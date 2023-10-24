import os
import json
import logging
import time
import numpy as np

from byte_mlperf.backends import runtime_backend
import byte_mlperf.datasets.fake_dataset.data_loader as fake_data_loader

import dlnne as nne
import pycuda.driver as cuda

log = logging.getLogger("BackendDL")

INPUT_TYPE = {
    "UINT8": np.uint8,
    "FLOAT32": np.float32,
    "LONG": np.long,
    "INT32": np.int32,
    "INT64": np.int64,
    "BOOL": np.bool
}

class DlModel:
    def __init__(self, engine, context, bindings):
        self.engine = engine
        self.context = context
        self.bindings = bindings

class Binding:
    def __init__(self, mem, name, is_input, np_type, shape, binding_size, max_batch_size, batch_binding_size, batch_binding_shape):
        self.mem = mem
        self.name = name
        self.is_input = is_input
        self.np_type = np_type
        self.shape = shape
        self.binding_size = binding_size
        self.max_batch_size = max_batch_size
        self.batch_binding_size = batch_binding_size
        self.batch_binding_shape = batch_binding_shape

    def __del__(self):
        self.mem.free()

    def get_element_size(self):
        return self.np_type(1).nbytes

    def get_element_count(self):
        element_count = 1
        for s in self.shape:
            element_count *= s
        return element_count

    def get_binding_size(self):
        return self.get_element_size() * self.get_element_count()

    def get_batch_binding_size(self):
        return self.max_batch_size * self.get_binding_size()

def nne_to_np_type(_type):
    if _type == nne.DataType.FLOAT:
        np_type = np.float32
    elif _type == nne.DataType.HALF:
        np_type = np.float16
    elif _type == nne.DataType.UINT8:
        np_type = np.uint8
    elif _type == nne.DataType.UINT16:
        np_type = np.uint16
    elif _type == nne.DataType.UINT32:
        np_type = np.uint32
    elif _type == nne.DataType.UINT64:
        np_type = np.uint64
    elif _type == nne.DataType.INT8:
        np_type = np.int8
    elif _type == nne.DataType.INT16:
        np_type = np.int16
    elif _type == nne.DataType.INT32:
        np_type = np.int32
    elif _type == nne.DataType.INT64:
        np_type = np.int64
    elif _type == nne.DataType.BOOL:
        np_type = np.int8
    else:
        print(_type)
        raise AssertionError("Unknown nne data type")
    return np_type

weight_share_configs = {
    "0": {
        "weight_mode": nne.WeightShareMode.single,
        "cluster_cfg": nne.ClusterConfig.cluser0,
    },
    "1": {
        "weight_mode": nne.WeightShareMode.single,
        "cluster_cfg": nne.ClusterConfig.cluser1,
    },
    "2": {
        "weight_mode": nne.WeightShareMode.single,
        "cluster_cfg": nne.ClusterConfig.cluser2,
    },
    "3": {
        "weight_mode": nne.WeightShareMode.single,
        "cluster_cfg": nne.ClusterConfig.cluser3,
    },
    "01": {
        "weight_mode": nne.WeightShareMode.share2,
        "cluster_cfg": nne.ClusterConfig.cluser01,
    },
    "23": {
        "weight_mode": nne.WeightShareMode.share2,
        "cluster_cfg": nne.ClusterConfig.cluser23,
    },
    "0123": {
        "weight_mode": nne.WeightShareMode.share4,
        "cluster_cfg": nne.ClusterConfig.cluser0123,
    },
}

class RuntimeBackendDL(runtime_backend.RuntimeBackend):
    def __init__(self):
        super(RuntimeBackendDL, self).__init__()
        log.info("INIT RuntimeBackendDL")
        self.hardware_type = 'DL'
        self.need_reload = False
        self.model_runtimes = []
        self.configs = None
        self.batch_size = -1

        cluster_count = cuda.get_cluster_count(0)
        if cluster_count == 1:
            self.weight_share = weight_share_configs["0"]
        elif cluster_count == 2:
            self.weight_share = weight_share_configs["01"]
        elif cluster_count == 4:
            self.weight_share = weight_share_configs["0123"]
        else:
            raise AssertionError("not match correct cluster count!")

    def predict_benchmark_init(self, test_data):
        if not self.model_runtimes:
            self.load()
        model_runtime = self.model_runtimes[0]
        bindings = model_runtime.bindings

        for binding in bindings:
            binding_name = binding.name
            binding_shape = binding.shape
            np_type = binding.np_type
            batch_binding_size = binding.batch_binding_size
            batch_binding_shape = binding.batch_binding_shape

            if binding.is_input:
                for input_name in test_data:
                    if input_name == binding_name:
                        hostmem = np.array(test_data[input_name]).astype(np_type)
                        cuda.memcpy_htod(binding.mem, hostmem)

        self.context = model_runtime.context
        self.binding_inputs = [binding.mem.as_buffer(binding.batch_binding_size) for binding in bindings]

    def predict_benchmark(self, execute_batch_size):
        self.context.execute(execute_batch_size, self.binding_inputs)
        
    def predict(self, feeds):
        if not self.model_runtimes:
            self.load()
        model_runtime = self.model_runtimes[0]
        engine = model_runtime.engine
        context = model_runtime.context

        batch_size = engine.max_batch_size

        bindings = model_runtime.bindings

        execute_batch_size = -1

        results = {}

        # h2d
        for binding in bindings:
            binding_name = binding.name
            binding_shape = binding.shape
            np_type = binding.np_type
            batch_binding_size = binding.batch_binding_size
            batch_binding_shape = binding.batch_binding_shape

            if binding.is_input:
                for input_name in feeds:
                    if input_name == binding_name:
                        hostmem = np.array(feeds[input_name]).astype(np_type)
                        cuda.memcpy_htod(binding.mem, hostmem)
                        if execute_batch_size == -1:
                            execute_batch_size = int(np.size(hostmem) / binding.get_element_count())
                            assert(execute_batch_size > 0 and execute_batch_size % 1 == 0)

        assert (execute_batch_size != -1)

        binding_inputs = [binding.mem.as_buffer(binding.batch_binding_size) for binding in bindings]
        context.execute(execute_batch_size, binding_inputs)

        # dth
        for binding in bindings:
            if not binding.is_input:
                output_name = binding.name
                output_shape = (binding.shape[0] * execute_batch_size, *binding.shape[1:])
                results[output_name]  = np.empty(output_shape, binding.np_type)
                cuda.memcpy_dtoh(results[output_name], binding.mem)
       
        return results

    def benchmark(self, dataloader):
        if not self.model_runtimes:
            self.load()
        iterations = self.workload['iterations']
        batch_size = self.get_loaded_batch_size()
        times_range = []
        report = {}
        report['BS'] = batch_size

        dataloader.rebatch(batch_size)
        if isinstance(dataloader, fake_data_loader.DataLoader):
            test_data = dataloader.get_samples(0)
        else:
            test_data, _ = dataloader.get_samples(0)
        self.predict_benchmark_init(test_data)

        for _ in range(30):
            self.predict_benchmark(batch_size)

        for _ in range(iterations):
            start_time = time.time()
            self.predict_benchmark(batch_size)
            end_time = time.time()
            times_range.append(end_time - start_time)

        times_range.sort()
        tail_latency = round(
            times_range[int(len(times_range) * 0.99)] * 1000, 2)
        avg_latency = round(sum(times_range) / iterations * 1000, 2)
        qps = int(1000.0 * batch_size / avg_latency)

        log.info(
            'Batch size is {}, QPS: {}, Avg Latency:{}, Tail Latency:{}'.
            format(batch_size, qps, avg_latency, tail_latency))

        report['QPS'] = qps
        report['AVG Latency'] = avg_latency
        report['P99 Latency'] = tail_latency

        return report

    def get_loaded_batch_size(self):
        return self.batch_size

    def load(self, batch_size) -> None:
        self.batch_size = batch_size
        self.model_runtimes = []
        self.input_type = self.configs['input_type']
        self.framework = self.configs['framework']

        self.model_name = self.configs['model']

        for i, segment in enumerate(self.configs['segments']):
            # there is no input/output meta data i the graph so it need to come from config.
            if not segment['input_tensor_map']:
                raise ValueError("Segment " + str(i) + " needs inputs")
            if not segment['output_tensor_map']:
                raise ValueError("Segment " + str(i) + " needs outputs")

            self.input_shapes = segment['input_tensor_map']
            self.outputs = segment['output_tensor_map'].split(",")

            engine_path = segment['compiled_model'][0]['compiled_obj']
            with open(engine_path, 'rb') as f:
                engine = nne.deserialize(f.read())

            assert(engine.max_batch_size == self.configs['max_batch_size'])

            context = engine.create_execution_context(self.weight_share['cluster_cfg'])

            num_bindings = engine.num_bindings
            max_batch_size = engine.max_batch_size
            bindings = []
            for index in range(num_bindings):
                binding_name = engine.get_binding_name(index)
                binding_is_input = engine.binding_is_input(index)
                np_type = nne_to_np_type(engine.get_binding_dtype(index))
                binding_shape = engine.get_binding_shape(index)

                binding_size = np_type(1).nbytes
                for s in binding_shape:
                    binding_size *= s

                batch_binding_size = max_batch_size * binding_size

                batch_binding_shape = (max_batch_size,) + binding_shape

                batch_mem = cuda.to_device(np.zeros(batch_binding_shape, dtype=np_type))

                binding = Binding(batch_mem, binding_name, binding_is_input, np_type, binding_shape, binding_size, engine.max_batch_size, batch_binding_size, batch_binding_shape)
                bindings.append(binding)

            self.model_runtimes.append(DlModel(engine, context, bindings))
