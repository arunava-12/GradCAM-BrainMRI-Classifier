import json
import zipfile
import os

def fix_keras_model(model_path, fixed_path):
    print(f"Opening {model_path}...")
    with zipfile.ZipFile(model_path, 'r') as zip_ref:
        config_data = zip_ref.read('config.json')
        config = json.loads(config_data)

    def remove_quantization_config(obj):
        if isinstance(obj, dict):
            if 'quantization_config' in obj:
                print(f"Removing quantization_config from {obj.get('name', 'unknown')}")
                del obj['quantization_config']
            for key, value in obj.items():
                remove_quantization_config(value)
        elif isinstance(obj, list):
            for item in obj:
                remove_quantization_config(item)

    print("Stripping quantization_config from layers...")
    remove_quantization_config(config)

    # Also check InputLayer issues if any
    # The error was: Unrecognized keyword arguments: ['batch_shape', 'optional']
    def fix_input_layer(obj):
        if isinstance(obj, dict):
            if obj.get('class_name') == 'InputLayer' or obj.get('class_name') == 'Input':
                c = obj.get('config', {})
                if 'batch_shape' in c:
                    # In some versions it's 'batch_input_shape'
                    pass
                if 'optional' in c:
                    print(f"Removing 'optional' from {obj.get('name', 'unknown')}")
                    del c['optional']
            for key, value in obj.items():
                fix_input_layer(value)
        elif isinstance(obj, list):
            for item in obj:
                fix_input_layer(item)

    fix_input_layer(config)

    print(f"Creating fixed model at {fixed_path}...")
    with zipfile.ZipFile(fixed_path, 'w') as new_zip:
        # Copy everything except config.json
        with zipfile.ZipFile(model_path, 'r') as old_zip:
            for item in old_zip.infolist():
                if item.filename != 'config.json':
                    new_zip.writestr(item, old_zip.read(item.filename))
        
        # Write the modified config.json
        new_zip.writestr('config.json', json.dumps(config))

    print("Done!")

if __name__ == "__main__":
    fix_keras_model("best_model_finetuned.keras", "best_model_fixed.keras")
