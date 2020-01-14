import os


scene_list_dir = '/home/wei/data/pdc/logs_wiper'
scene_list_prefix = 'logs_wiper/'


def run_make_list():
    assert os.path.exists(scene_list_dir)
    log_items = []
    for subdir in os.listdir(scene_list_dir):
        fullpath_subdir = os.path.join(scene_list_dir, subdir)
        if os.path.isdir(fullpath_subdir):
            log_items.append(scene_list_prefix + subdir)

    # The output
    outfile = open('built_log.txt', 'w')
    for item in log_items:
        outfile.write(item)
        outfile.write('\n')


if __name__ == '__main__':
    run_make_list()
