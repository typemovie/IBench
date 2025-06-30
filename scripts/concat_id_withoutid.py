import json


def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def merge_json_data(json1, json2):
    merged_data = {'datas': []}

    for data1 in json1['datas']:
        for data2 in json2['datas']:
            if data1['id'] == data2['id'] and data1['prompt'] == data2['prompt']:
                merged_entry = {
                    "num": data1['num'],
                    "id": data1['id'],
                    "prompt": data1['prompt'],
                    "prompt_attr": data1['prompt_attr'],
                    "prompt_style": data1['prompt_style'],
                    "imagewithid": data1['imagewithid'],
                    "imagewithoutid": data2['imagewithoutid']
                }
                merged_data['datas'].append(merged_entry)

    return merged_data


def save_json(data, file_path):
    """
    Save JSON data to a file.

    :param data: JSON data to save
    :param file_path: Path to the output JSON file
    """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    print(f'Merged JSON saved as {file_path}')


if __name__ == '__main__':
    config1 = {
        "input_file_1": r"D:\common_tools\Blade\evaluation\imagegen\imagewithid_v1_short.json",
        "input_file_2": r"D:\common_tools\Blade\evaluation\imagewithoutid_t2i_short.json",
        "output_file": r"D:\common_tools\IBench\data\unsplash50_short_v1.json"
    }

    config2 = {
        "input_file_1": r"D:\common_tools\Blade\evaluation\imagegen\imagewithid_sdxl_short.json",
        "input_file_2": r"D:\common_tools\Blade\evaluation\imagewithoutid_sdxl_t2i_short.json",
        "output_file": r"D:\common_tools\IBench\data\unsplash50_short_sdxl.json"
    }

    config3 = {
        "input_file_1": r"D:\common_tools\Blade\evaluation\imagegen\imagewithid_instantid_short.json",
        "input_file_2": r"D:\common_tools\Blade\evaluation\imagewithoutid_t2i_short.json",
        "output_file": r"D:\common_tools\IBench\data\unsplash50_short_sdxl_instantid.json"
    }

    config4 = {
        "input_file_1": r"D:\common_tools\Blade\evaluation\imagegen\imagewithid_v11_short.json",
        "input_file_2": r"D:\common_tools\Blade\evaluation\imagewithoutid_t2i_short.json",
        "output_file": r"D:\common_tools\IBench\data\unsplash50_short_v11.json"
    }

    config5 = {
        "input_file_1": r"D:\common_tools\Blade\evaluation\imagegen\imagewithid_v12_short.json",
        "input_file_2": r"D:\common_tools\Blade\evaluation\imagewithoutid_t2i_short.json",
        "output_file": r"D:\common_tools\IBench\data\unsplash50_short_v12.json"
    }

    config6 = {
        "input_file_1": r"D:\common_tools\Blade\evaluation\imagegen\imagewithid_v1_chineseid_longer.json",
        "input_file_2": r"D:\common_tools\Blade\evaluation\imagewithoutid_t2i_chinese_longer.json",
        "output_file": r"D:\common_tools\IBench\data\chineseid_longer_v1.json"
    }

    config7 = {
        "input_file_1": r"D:\common_tools\Blade\evaluation\imagegen\imagewithid_v11_chineseid_longer.json",
        "input_file_2": r"D:\common_tools\Blade\evaluation\imagewithoutid_t2i_chinese_longer.json",
        "output_file": r"D:\common_tools\IBench\data\chineseid_longer_v11.json"
    }

    config8 = {
        "input_file_1": r"D:\common_tools\Blade\evaluation\imagegen\imagewithid_v12_chineseid_longer.json",
        "input_file_2": r"D:\common_tools\Blade\evaluation\imagewithoutid_t2i_chinese_longer.json",
        "output_file": r"D:\common_tools\IBench\data\chineseid_longer_v12.json"
    }

    config9 = {
        "input_file_1": r"D:\common_tools\Blade\evaluation\imagegen\imagewithid_v1_generateid_typemovie.json",
        "input_file_2": r"D:\common_tools\Blade\evaluation\imagewithoutid_t2i_generateid_typemovie.json",
        "output_file": r"D:\common_tools\IBench\data\generateid_typemovie_v1.json"
    }

    config10 = {
        "input_file_1": r"D:\common_tools\Blade\evaluation\imagegen\imagewithid_v12_generateid_typemovie.json",
        "input_file_2": r"D:\common_tools\Blade\evaluation\imagewithoutid_t2i_generateid_typemovie.json",
        "output_file": r"D:\common_tools\IBench\data\generateid_typemovie_v12.json"
    }

    config11 = {
        "input_file_1": r"D:\common_tools\Blade\evaluation\imagegen\imagewithid_v11_generateid_typemovie.json",
        "input_file_2": r"D:\common_tools\Blade\evaluation\imagewithoutid_t2i_generateid_typemovie.json",
        "output_file": r"D:\common_tools\IBench\data\generateid_typemovie_v11.json"
    }

    config12 = {
        "input_file_1": r"D:\common_tools\Blade\evaluation\imagegen\imagewithid_instantid_chineseid_longer.json",
        "input_file_2": r"D:\common_tools\Blade\evaluation\imagewithoutid_sdxl_t2i_chinese_longer.json",
        "output_file": r"D:\common_tools\IBench\data\chineseid_longer_sdxl_instantid.json"
    }

    config13 = {
        "input_file_1": r"D:\common_tools\Blade\evaluation\imagegen\imagewithid_instantid_generateid_typemovie.json",
        "input_file_2": r"D:\common_tools\Blade\evaluation\imagewithoutid_sdxl_t2i_generateid_typemovie.json",
        "output_file": r"D:\common_tools\IBench\data\generateid_typemovie_sdxl_instantid.json"
    }

    config14 = {
        "input_file_1": r"D:\common_tools\Blade\evaluation\imagegen\imagewithid_sdxl_generateid_typemovie.json",
        "input_file_2": r"D:\common_tools\Blade\evaluation\imagewithoutid_sdxl_t2i_generateid_typemovie.json",
        "output_file": r"D:\common_tools\IBench\data\generateid_typemovie_sdxl_pulid.json"
    }

    config15 = {
        "input_file_1": r"D:\common_tools\Blade\evaluation\imagegen\imagewithid_sdxl_chineseid_longer.json",
        "input_file_2": r"D:\common_tools\Blade\evaluation\imagewithoutid_sdxl_t2i_chinese_longer.json",
        "output_file": r"D:\common_tools\IBench\data\chineseid_longer_sdxl_pulid.json"
    }

    config16 = {
        "input_file_1": r"D:\common_tools\Blade\evaluation\imagegen\imagewithid_v12_preview_0212_chineseid_longer.json",
        "input_file_2": r"D:\common_tools\Blade\evaluation\imagewithoutid_t2i_chinese_longer.json",
        "output_file": r"D:\common_tools\IBench\data\chineseid_longer_v12_preview_0212.json"
    }

    config17 = {
        "input_file_1": r"D:\common_tools\Blade\evaluation\imagegen\imagewithid_v12_preview_0213_chineseid_longer.json",
        "input_file_2": r"D:\common_tools\Blade\evaluation\imagewithoutid_t2i_chinese_longer.json",
        "output_file": r"D:\common_tools\IBench\data\chineseid_longer_v12_preview_0213.json"
    }

    config18 = {
        "input_file_1": r"D:\common_tools\Blade\evaluation\imagegen\imagewithid_v12_preview_0217_chineseid_longer.json",
        "input_file_2": r"D:\common_tools\Blade\evaluation\imagewithoutid_t2i_chinese_longer.json",
        "output_file": r"D:\common_tools\IBench\data\chineseid_longer_v12_preview_0217.json"
    }

    config19 = {
        "input_file_1": r"D:\common_tools\Blade\evaluation\imagegen\imagewithid_v12_preview_0218_chineseid_longer.json",
        "input_file_2": r"D:\common_tools\Blade\evaluation\imagewithoutid_t2i_chinese_longer.json",
        "output_file": r"D:\common_tools\IBench\data\chineseid_longer_v12_preview_0218.json"
    }

    config20 = {
        "input_file_1": r"D:\common_tools\Blade\evaluation\imagegen\imagewithid_v12_preview_0219_chineseid_longer.json",
        "input_file_2": r"D:\common_tools\Blade\evaluation\imagewithoutid_t2i_chinese_longer.json",
        "output_file": r"D:\common_tools\IBench\data\chineseid_longer_v12_preview_0219.json"
    }

    config21 = {
        "input_file_1": r"D:\common_tools\Blade\evaluation\imagegen\imagewithid_v12_preview_0220_chineseid_longer.json",
        "input_file_2": r"D:\common_tools\Blade\evaluation\imagewithoutid_t2i_chinese_longer.json",
        "output_file": r"D:\common_tools\IBench\data\chineseid_longer_v12_preview_0220.json"
    }

    config22 = {
        "input_file_1": r"D:\common_tools\Blade\evaluation\imagegen\imagewithid_v12_preview_0221_chineseid_longer.json",
        "input_file_2": r"D:\common_tools\Blade\evaluation\imagewithoutid_t2i_chinese_longer.json",
        "output_file": r"D:\common_tools\IBench\data\chineseid_longer_v12_preview_0221.json"
    }

    config23 = {
        "input_file_1": r"D:\common_tools\Blade\evaluation\imagegen\imagewithid_v12_preview_0222_chineseid_longer.json",
        "input_file_2": r"D:\common_tools\Blade\evaluation\imagewithoutid_t2i_chinese_longer.json",
        "output_file": r"D:\common_tools\IBench\data\chineseid_longer_v12_preview_0222.json"
    }

    config24 = {
        "input_file_1": r"D:\common_tools\Blade\evaluation\imagegen\imagewithid_v12_preview_0223_chineseid_longer.json",
        "input_file_2": r"D:\common_tools\Blade\evaluation\imagewithoutid_t2i_chinese_longer.json",
        "output_file": r"D:\common_tools\IBench\data\chineseid_longer_v12_preview_0223.json"
    }

    config25 = {
        "input_file_1": r"D:\common_tools\Blade\evaluation\imagegen\imagewithid_v12_preview_0224_chineseid_longer.json",
        "input_file_2": r"D:\common_tools\Blade\evaluation\imagewithoutid_t2i_chinese_longer.json",
        "output_file": r"D:\common_tools\IBench\data\chineseid_longer_v12_preview_0224.json"
    }

    config26 = {
        "input_file_1": r"D:\common_tools\Blade\evaluation\imagegen\imagewithid_v12_preview_0225_chineseid_longer.json",
        "input_file_2": r"D:\common_tools\Blade\evaluation\imagewithoutid_t2i_chinese_longer.json",
        "output_file": r"D:\common_tools\IBench\data\chineseid_longer_v12_preview_0225.json"
    }

    config27 = {
        "input_file_1": r"D:\common_tools\Blade\evaluation\imagegen\imagewithid_v12_preview_0226_chineseid_longer.json",
        "input_file_2": r"D:\common_tools\Blade\evaluation\imagewithoutid_t2i_chinese_longer.json",
        "output_file": r"D:\common_tools\IBench\data\chineseid_longer_v12_preview_0226.json"
    }

    config28 = {
        "input_file_1": r"D:\common_tools\Blade\evaluation\imagegen\imagewithid_v12_preview_0227_chineseid_longer.json",
        "input_file_2": r"D:\common_tools\Blade\evaluation\imagewithoutid_t2i_chinese_longer.json",
        "output_file": r"D:\common_tools\IBench\data\chineseid_longer_v12_preview_0227.json"
    }

    config29 = {
        "input_file_1": r"D:\common_tools\Blade\evaluation\imagegen\imagewithid_v12_preview_0228_chineseid_longer.json",
        "input_file_2": r"D:\common_tools\Blade\evaluation\imagewithoutid_t2i_chinese_longer.json",
        "output_file": r"D:\common_tools\IBench\data\chineseid_longer_v12_preview_0228.json"
    }

    config30 = {
        "input_file_1": r"D:\common_tools\Blade\evaluation\imagegen\imagewithid_v12_preview_0229_chineseid_longer.json",
        "input_file_2": r"D:\common_tools\Blade\evaluation\imagewithoutid_t2i_chinese_longer.json",
        "output_file": r"D:\common_tools\IBench\data\chineseid_longer_v12_preview_0229.json"
    }

    try:
        config = config29
        input_file_1 = config['input_file_1']
        input_file_2 = config['input_file_2']
        output_file = config['output_file']

        # Load JSON files
        json1 = load_json(input_file_1)
        json2 = load_json(input_file_2)

        # Merge JSON data
        merged_json = merge_json_data(json1, json2)

        # Save the merged JSON data
        save_json(merged_json, output_file)
    except Exception as e:
        print(f"An error occurred: {e}")
