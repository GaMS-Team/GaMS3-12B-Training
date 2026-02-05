def get_data_parameters():
    paths = [
        0.063,
        "/data/DGT_text_document",
        0.002,
        "/data/KAS_text_document",
        0.034,
        "/data/macocu_text_document",
        0.901,
        "/data/wikipedia_en_sl_translated_text_document"
    ]
    split = "9900,70,30"
    index_mapping_dir = "/data/12b_train_idx"

    param_dict = {
        "paths": paths,
        "split": split,
        "index_mapping_dir": index_mapping_dir
    }

    return param_dict