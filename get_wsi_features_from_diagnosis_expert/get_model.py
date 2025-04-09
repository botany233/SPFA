import os
import csv
import shutil

input_model_dir = "/home/chengfangchi/graduate2302/code/cfc/最后的冯如杯/transmil/record/simple_mix"
choose_value = ["val_acc", "val_macro_f1"][0]

output_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

for model_name in os.listdir(input_model_dir):
    model_dir = os.path.join(input_model_dir, model_name)
    with open(os.path.join(model_dir, "record.csv"), "r") as f:
        data = list(csv.reader(f))
    value_datas, titles = data[1:], data[0]
    titles = [i.strip(" ") for i in titles]
    model_epoch, max_value = -1, 0
    for i in value_datas:
        epoch, value = int(i[titles.index("epoch")]), float(i[titles.index(choose_value)])
        if max_value <= value:
            max_value, model_epoch = value, epoch
    
    input_model_path = os.path.join(model_dir, "model", str(model_epoch) + ".pickle")
    output_model_path = os.path.join(output_model_dir, model_name + ".pickle")

    shutil.copy(input_model_path, output_model_path)