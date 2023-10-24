import os
import json
import logging
import time
import numpy as np
import subprocess
from typing import Any, Dict
from pathlib import Path, PosixPath
from tools import saved_to_onnx, torch_to_onnx
import byte_mlperf.datasets.fake_dataset.data_loader as fake_data_loader


from byte_mlperf.backends import compile_backend

from os.path import splitext, basename, join

from dl.quantize.quantize import quantize, qconfig
from tvm import relay
from tvm import topi
from tvm.driver import tvmc
import tvm
import dlnne as nne
import pycuda.driver as cuda
from pydlnne_modulator import *

log = logging.getLogger("CompileBackendDL")


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

def get_params_info(model_path, params=None):
    tvm_model = tvmc.frontends.load_model(model_path)
    mod, params = tvm_model if isinstance(tvm_model, tuple) else (tvm_model.mod, tvm_model.params)
    if params:
        mod['main'] = relay.build_module.bind_params_by_name(mod['main'], params)
        func = relay.build_module.bind_params_by_name(mod["main"], params)
    else:
        func = mod["main"]
    inputs_name = []
    inputs_shape = []
    inputs_np_data_type = []
    for v in func.params:
        inputs_name.append(v.name_hint)
        inputs_shape.append(topi.utils.get_const_tuple(v.type_annotation.shape))
        inputs_np_data_type.append(v.type_annotation.dtype)
    return inputs_np_data_type, inputs_name, inputs_shape, mod, params

def Get_IBuildModulatorImpl(split_nodes):
    class IBuildModulatorImpl(IBuildModulator):
        def ScheduleDAG(self, graph):
            return False

        def GenerateSubgraph(self, graph, container):
            name_node_map = {}
            sg_node_list = []
            node_list = graph.GetNodes()
            partition_config=[]
            partition_tmp = []
            current_config_index = 0
            print(split_nodes)
            for node in node_list:
                name_node_map[node.GetName()] = node
                partition_tmp.append(node.GetName())
                if node.GetName() == split_nodes[current_config_index]:
                    partition_config.append(partition_tmp)
                    partition_tmp = []
                    current_config_index += 1
            if len(partition_tmp) > 0:        
                partition_config.append(partition_tmp)

            for config in partition_config:
                print("new subgraph: nodes count={}".format(len(config)))
                sg_node_list.clear()

                for node_name in config:
                    if name_node_map.get(node_name) is None:
                        raise Exception('Error: node does not exist.')
                    sg_node_list.append(name_node_map[node_name])
                container.AddSubgraph(sg_node_list)
            return True
    return IBuildModulatorImpl()



class CompileBackendDL(compile_backend.CompileBackend):
    def __init__(self):
        super(CompileBackendDL, self).__init__()
        log.info("INIT CompileBackendDL")
        self.hardware_type = 'DL'
        self.model_config = None
        self.dl_interact_info_config = {}
        self.current_dir = os.path.split(os.path.abspath(__file__))[0]

        cluster_count = cuda.get_cluster_count(0)
        if cluster_count == 1:
            self.weight_share = weight_share_configs["0"]
        elif cluster_count == 2:
            self.weight_share = weight_share_configs["01"]
        elif cluster_count == 4:
            self.weight_share = weight_share_configs["0123"]
        else:
            raise AssertionError("not match correct cluster count!")


    def pre_optimize(self, configs: Dict[str, Any]):
        """Model pre-optimization interface.

        Requirements: Model pre-optimization
        cannot change the model format. Torch model export to ONNX is allowed.
        """
        model_info = configs["model_info"]
        model_type = model_info["model_format"]
        model_name = model_info["model"]

        pre_optimized_root = Path(self.current_dir) / "pre_optimized_models"
        if not pre_optimized_root.exists():
            pre_optimized_root.mkdir(parents=True)

        model_path = os.path.abspath(configs["model_info"]["model_path"])
        onnx_path = pre_optimized_root / (model_name + ".onnx")
        if not self.model_config:
            self.model_config = configs.get("interact_info", {})

        # convert model to onnx if it's not
        # configs['workload'] is the content of workloads/<task_name>.json and
        # configs['model_info'] is content of model_zoo/<task_name>.json
        if model_type != "onnx":
            if onnx_path.exists():
                # onnx_path = self._update_pack_model(onnx_path, model_info)
                model_info["model_path"] = onnx_path
                log.info("{} file exists, skip ONNX conversion".format(onnx_path.name))
            else:
                # convert the model to onnx
                log.info(
                    "Convert the model: {} from format: {} to onnx".format(
                        model_name, model_type
                    )
                )
                if model_type == "saved_model":
                    saved_to_onnx.savedmodel_to_onnx(model_path, onnx_path)
                    # onnx_path = self._update_pack_model(onnx_path, model_info)
                elif model_type == "pt":
                    torch_to_onnx.torch_to_onnx(model_path, str(onnx_path))
                    # onnx_path = self._update_pack_model(onnx_path, model_info)
                else:
                    log.error(
                        "Wrong model type: {}, which must be saved_model, pt, or onnx".format(
                            model_type
                        )
                    )
                    raise TypeError("Model type must be saved_model, pt, or onnx")

                if os.path.exists(onnx_path):
                    model_info["model_path"] = onnx_path
                    log.info(
                        "Converted the model: {} from format: {} to onnx".format(
                            model_name, model_type
                        )
                    )
                else:
                    log.error(
                        "{} not exists, failed to convert the model: {} to onnx".format(
                            onnx_path, model_name
                        )
                    )
                    raise RuntimeError("Failed to convert model to onnx")
        else:
            log.info("{} is onnx model, skip ONNX conversion".format(model_name))

        if isinstance(model_info["model_path"], PosixPath):
            model_info["model_path"] = str(model_info["model_path"])

        return configs

    def compile(self, config, dataloader=None):
        interact_info_file = os.path.join(
            self.current_dir, "interact_infos", config["model_info"]["model"] + ".json"
        )
        if os.path.exists(interact_info_file):
             with open(interact_info_file, "r") as f:
                self.dl_interact_info_config = json.load(f)
        

        interact_info = config.get("interact_info", {})
        
        log.info("dl_interact_info_config: {}".format(json.dumps(self.dl_interact_info_config, indent=4, separators=(', ', ': '), ensure_ascii=False)))

        osEnv = self.dl_interact_info_config.get("osEnv", {})
        for k,v in osEnv.items():
            os.environ[k] = v


        precision = self.dl_interact_info_config.get("precision", "no_quantize")

        os_env_model_precision = os.environ.get('DL_BYTEMLPERF_MODEL_PRECISION')
        if os_env_model_precision:
            log.info("DL_BYTEMLPERF_MODEL_PRECISION has value, override model precision from {} to {}".format(precision, os_env_model_precision))
            precision = os_env_model_precision

        is_recompile = False if os.environ.get('DL_BYTEMLPERF_IS_RECOMPILE_MODEL') == "0" else True

        # other model format to .rlym 
        dl_models_dir = self.current_dir + "/dl_models"

        if not os.path.exists(dl_models_dir):
            os.mkdir(dl_models_dir)

        model_path = config['model_info']['model_path']

        python3_root_dir = ""

        converted_model_path = dl_models_dir + "/" + splitext(basename(model_path))[0] + ".converted.rlym"

        convert_model_shell_command = "python3 -m dl convert "

        input_shape_command = "--input-shapes \""
        for input_name in config['model_info']['input_shape']:
            input_shape = config['model_info']['input_shape'][input_name]
            input_shape_str = ",".join(str(e) for e in input_shape)

            input_shape_command = input_shape_command + input_name + ":[" + input_shape_str + "]"
        input_shape_command += "\""

        convert_model_shell_command += input_shape_command
        convert_model_shell_command += " --output-model " + converted_model_path + " " + model_path
        convert_model_shell_command = python3_root_dir + convert_model_shell_command

        if is_recompile or not os.path.exists(converted_model_path):
            log.info("converting model to rlym: {}".format(convert_model_shell_command))
            os.system(convert_model_shell_command)

        config['model_info']['model_path'] = converted_model_path

        
        tvmQuantizeArgs = self.dl_interact_info_config.get("tvmQuantizeArgs", "")


        # fp16 dowmcat
        if precision == "fp16_downcast":
            log.info("quantize model with fp16 downcast")
            downcast_model_path = dl_models_dir + "/" + splitext(basename(model_path))[0] + ".converted.fp16.rlym"
            downcast_fp16_command = "python3 -m dl quantize " + tvmQuantizeArgs
            downcast_fp16_command = downcast_fp16_command + " --downcast " + converted_model_path + " --output-model " + downcast_model_path
            downcast_fp16_command = python3_root_dir + downcast_fp16_command

            if is_recompile or not os.path.exists(downcast_model_path):
                log.info("downcast fp16 model to rlym: {}".format(downcast_fp16_command))
                os.system(downcast_fp16_command)
            config['model_info']['model_path'] = downcast_model_path
            



        # quantize to int8
        if dataloader is not None and precision == "int8":
            log.info("quantize model with int8")

            # Save the Model
            model_name = splitext(basename(converted_model_path))[0] + ".int8.rlym"
            quantized_model_path = dl_models_dir + "/" + model_name

            if is_recompile or not os.path.exists(quantized_model_path):
                inputs_data_type, inputs_name, inputs_shape, mod, params = get_params_info(converted_model_path)

                datasets = dataloader.batched_data
                if isinstance(dataloader, fake_data_loader.DataLoader):
                    datasets = []
                    for i in range(dataloader.get_batch_count()):
                        data_ = dataloader.get_samples(i)
                        datasets.append(data_)
                
                with qconfig():
                    mod = quantize(mod, params, datasets)
                with open(quantized_model_path, "w") as f:
                    f.write(tvm.ir.save_json(mod))
                    log.info("Quantized model is saved in {}".format(quantized_model_path))

            config['model_info']['model_path'] = quantized_model_path


        
        # build engine
        callback = None
        if self.dl_interact_info_config and self.dl_interact_info_config.get("sub_graph") is not None:
            sub_graph_config = self.dl_interact_info_config.get("sub_graph")
            if sub_graph_config.get(precision) is not None:
                sub_graph_nodes = sub_graph_config.get(precision)
                if len(sub_graph_nodes) > 0:
                    log.info("register sub graph nodes: {}".format(sub_graph_nodes))
                    callback = Get_IBuildModulatorImpl(sub_graph_nodes)

        networkConfig = self.dl_interact_info_config.get("networkConfig", "")

        max_batch_size = config['model_info']['max_batch_size'] if config['model_info']['max_batch_size'] > 1 else 32

        engine_file_name = splitext(basename(config['model_info']['model_path']))[0] + ".slz"
        engine_file_path = dl_models_dir + "/" + engine_file_name
        if is_recompile or not os.path.exists(engine_file_path):
            with nne.Builder() as builder, nne.Parser() as parser:
                network = builder.create_network()
                log.info("networkConfig: {}".format(networkConfig))
                if networkConfig:
                    log.info("networkConfig work! networkConfig: {}".format(networkConfig))
                    network.set_config(networkConfig)
                weight_mode = self.weight_share['weight_mode']
                builder.config.ws_mode = weight_mode
                builder.config.max_batch_size = max_batch_size
                if callback:
                    log.info("callback work!")
                    builder.config.callback = callback

                parser.parse(config['model_info']['model_path'], network)

                with builder.build_engine(network) as engine:
                    with open(engine_file_path, 'wb') as f:
                        f.write(engine.serialize())

        config['model_info']['model_path'] = engine_file_path

        self.dl_interact_info_config.pop("precision")
        optimizations_config = self.dl_interact_info_config

        result = {
            "model": config['model_info']['model'],
            "framework": config['model_info']['framework'],
            "compile_precision": precision.upper(),
            "optimizations": optimizations_config,
            "instance_count": 1,
            "device_count": 1,
            "input_type": config['model_info']['input_type'].split(","),
            "max_batch_size": config['model_info']['max_batch_size'],
            "compile_status": "success",
            "sg_percent": 100,
            "segments": [
                {
                    "sg_idx": 0,
                    "is_fallback": False,
                    "input_tensor_map": config['model_info']['input_shape'],
                    "output_tensor_map":config['model_info']['outputs'],
                    "compiled_model": [
                        {
                            "compiled_bs": max_batch_size,
                            "compiled_obj": config['model_info']['model_path'],
                        },
                    ],
                },
            ]
        }

        self.configs = result
        self.workload = config['workload']
        self.model_info = config['model_info']
        return result

    def get_interact_profile(self, config):
        model_profile = []
        file_path = "byte_mlperf/backends/DL/" + self.hardware_type + '.json'
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                model_profile = json.load(f)
        else:
            log.info(
                'File path: {} does not exist, please check'.format(file_path))

        

        return model_profile

    def get_best_batch_size(self):
        """
        Get Best Batch Size for the model
        """
        return None
