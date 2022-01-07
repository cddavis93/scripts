#!/usr/bin/env python3

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

fname = sys.argv[1]

header = ["Name", "Self CPU %", "Self CPU", "CPU total %", "CPU total", "CPU time avg", "Self CUDA", "Self CUDA %", "CUDA total", "CUDA time avg", "# of Calls", "Input Shapes"]
percent_cols =  ["Self CPU %", "CPU total %", "Self CUDA %"]
time_cols = ["Self CPU", "CPU total", "CPU time avg", "Self CUDA", "CUDA total", "CUDA time avg"]

function_classes = {\
    "Dense MM": {"aten::mm", "aten::addmm", "aten::bmm", "aten::linear", "aten::matmul"},\
    "Sparse MM": {"aten::_sparse_addmm"},\
    "Convolution": {"aten::conv2d", "aten::convolution", "aten::_convolution", "aten::cudnn_convolution", "aten::conv_transpose2d", "aten::cudnn_convolution_transpose", "aten::thnn_conv_depthwise2d", "aten::thnn_conv_depthwise2d_forward"},\
    "Element-wise Ops": {"aten::layer_norm", "aten::native_layer_norm", "aten::softmax", "aten::relu", "aten::_softmax", "aten::log_softmax", "aten::_log_softmax", "aten::sub", "aten::floor_divide_", "aten::mul", "aten::sum", "aten::add", "aten::div", "aten::floor_divide", "aten::lt", "aten::gt", "aten::ge", "aten::add_", "aten::is_nonzero", "aten::bitwise_not", "aten::bitwise_or_", "aten::bitwise_or", "aten::any", "aten::max", "aten::ne", "aten::eq", "aten::threshold", "aten::fmod_", "aten::batch_norm", "aten::_batch_norm_impl_index", "aten::cudnn_batch_norm", "aten::isfinite", "aten::clamp", "aten::relu_", "aten::abs", "aten::threshold_", "aten::all", "aten::bitwise_and", "aten::__and__", "aten::sigmoid", "aten::mul_", "aten::clamp_max", "aten::maximum", "aten::minimum", "aten::exp", "aten::log", "aten::min", "aten::clamp_min", "aten::rsub"},\
    "Regional": {"aten::adaptive_avg_pool2d", "torchvision::nms", "torchvision::roi_align", "aten::max_pool2d_with_indices", "aten::max_pool2d", "aten::argmax", "aten::norm", "PrRoIPool2DFunction"},\
    "Embedding": {"aten::embedding", "aten::slice"},\
    "Data Movement": {"aten::to", "aten::cat", "aten::select", "aten::index_put_", "aten::_index_put_impl_", "aten::_cat", "aten::copy_", "aten::clone", "aten::masked_fill", "aten::masked_fill_", "aten::index_select", "aten::contiguous", "aten::repeat", "aten::stack", "aten::fill_", "aten::set_", "aten::zero_"},\
    "Data Transformation": {"aten::coalesce",  "aten::transpose", "aten::transpose_", "aten::reshape", "aten::narrow", "aten::arange", "aten::permute", "aten::unsqueeze_", "aten::unsqueeze", "aten::as_strided_", "aten::as_strided", "aten::masked_select", "aten::nonzero", "aten::unfold", "aten::t_", "aten::t"}
}

fallback_class = "Other"
function_types = {}
for class_name, class_fs in function_classes.items():
    for f in class_fs:
        function_types[f] = class_name

has_tensor_shapes = True

def time_str_to_float(t):
    if t[-2:] == "ms":
        return float(t[:-2]) / 1000
    elif t[-2:] == "us":
        return float(t[:-2]) / 1000000
    elif t[-2:] == "ns":
        return float(t[:-2]) / 1000000000
    else: # seconds?
        return float(t[:-1])

def time_float_to_str(t):
    units = ["s", "ms", "us", "ns"]
    unit_idx = 0
    while (t < 1) and (unit_idx < len(units)-1):
        t *= 1000
        unit_idx += 1
    return f"{round(t,1)}{units[unit_idx]}"


with open(fname, "r") as f:
    lines = f.read().split("\n")

found_start = False
for line_idx, line in enumerate(lines):
    if line.startswith("--------"): # bit hacky...
        found_start = True
        start_idx = line_idx
        break
if not found_start:
    print("Could not find profiler results...")
    sys.exit()

fields = []
for line in lines[start_idx+3:]:
    if line.startswith("--------"):
        break
    line_fields = line.split(" ")
    line_fields = [f for f in line_fields if len(f) > 0]
    if len(line_fields) < len(header):
        print("Data has fewer columns than expected; no tensor shapes?")
        has_tensor_shapes = False
        header = header[:len(line_fields)]
    line_fields = line_fields[0:len(header)-1] + [" ".join(line_fields[len(header)-1:])]
    fields.append(line_fields)

#import pdb; pdb.set_trace()
df = pd.DataFrame(fields, columns=header)

for col_name in percent_cols:
    df[col_name] = df[col_name].apply(lambda x: float(x[:-1]))

for col_name in time_cols:
    df[col_name] = df[col_name].apply(lambda x: time_str_to_float(x))

unknown_names = set()
cpu_runtimes = {k: 0 for k in function_classes.keys()}
cpu_runtimes[fallback_class] = 0
#cpu_runtimes["Toplevel"] = df.iloc[0]["Self CPU"]
gpu_runtimes = {k: 0 for k in function_classes.keys()}
gpu_runtimes[fallback_class] = 0
#gpu_runtimes["Toplevel"] = df.iloc[0]["Self CUDA"]

for i in range(0, df.shape[0]):
    row = df.iloc[i]
    name = row["Name"]
    if name in function_types:
        fclass = function_types[name]
    else:
        if name not in unknown_names:
            print(f"Unknown fname {name}")
            unknown_names.add(name)
        fclass = fallback_class
    cpu_runtimes[fclass] += row["Self CPU"]
    gpu_runtimes[fclass] += row["Self CUDA"]
print("Actual runtimes:")
print(f"CPU: {cpu_runtimes}")
print(f"GPU: {gpu_runtimes}")
total_cpu = sum(cpu_runtimes.values())
total_gpu = sum(gpu_runtimes.values())
cpu_fracts = {}
gpu_fracts = {}
for k in cpu_runtimes.keys():
    cpu_fracts[k] = cpu_runtimes[k] / total_cpu
for k in gpu_runtimes.keys():
    gpu_fracts[k] = gpu_runtimes[k] / total_gpu
print("As portion of whole:")
print(f"CPU: {cpu_fracts}")
print(f"GPU: {gpu_fracts}")

chart_labels = list(function_classes.keys()) + [fallback_class]

plt.clf()
plt.pie([cpu_runtimes[l] for l in chart_labels], normalize=True, startangle=90, counterclock=False, labels=None, autopct=(lambda x: ""))
plt.subplots_adjust(left=0.1, bottom=0.1, right=7.5)
plt.legend(loc="center right", labels=[f"{l}, {time_float_to_str(cpu_runtimes[l])} ({round(100*cpu_fracts[l],1)}%)" for l in chart_labels],
    bbox_transform=plt.gcf().transFigure, bbox_to_anchor=(1,0.5), fontsize=9)
plt.subplots_adjust(left=0.0, bottom=0.0, right=0.6)
cpu_fname = os.path.splitext(fname)[0] + "_cpu.png"
plt.savefig(cpu_fname, bbox_inches="tight")

plt.clf()
plt.pie([gpu_runtimes[l] for l in chart_labels], normalize=True, startangle=90, counterclock=False, labels=None, autopct=(lambda x: ""))
plt.subplots_adjust(left=0.1, bottom=0.1, right=7.5)
plt.legend(loc="center right", labels=[f"{l}, {time_float_to_str(gpu_runtimes[l])} ({round(100*gpu_fracts[l],1)}%)" for l in chart_labels],
    bbox_transform=plt.gcf().transFigure, bbox_to_anchor=(1,0.5), fontsize=9)
plt.subplots_adjust(left=0.0, bottom=0.0, right=0.6)
gpu_fname = os.path.splitext(fname)[0] + "_gpu.png"
plt.savefig(gpu_fname, bbox_inches="tight")
