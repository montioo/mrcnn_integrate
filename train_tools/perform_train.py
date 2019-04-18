import os


MASKRCNN_BENCHMARK_DIR = '/home/wei/Coding/mrcnn/maskrcnn-benchmark'
train_executable = os.path.join(MASKRCNN_BENCHMARK_DIR, "tools", "train_net.py")
cfg_file = '/home/wei/Coding/mrcnn/mrcnn_integrate/config/e2e_mask_rcnn_R_50_FPN_1x_boot.yaml'
output_dir = '/home/wei/Coding/mrcnn/mrcnn_integrate/dataproc/tmp/output'
paths_catalog = '/home/wei/Coding/mrcnn/mrcnn_integrate/train_tools/paths_catalog_pdc.py'

args = ""

args += "OUTPUT_DIR \"%s\" " % (output_dir)
args += "PATHS_CATALOG \"%s\" " % (paths_catalog)

cmd = "python %s --config-file %s %s" % (train_executable, cfg_file, args)
cmd = cmd + ' SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 720000 SOLVER.STEPS "(480000, 640000)" TEST.IMS_PER_BATCH 1'
print(cmd)

# copy file to output directory
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

if __name__ == '__main__':
    os.system(cmd)
