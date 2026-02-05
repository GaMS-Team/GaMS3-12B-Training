def get_data_parameters():
    paths = [
        0.054,
        "/data/nemotron_data/math_4_plus_text_document",
        0.061,
        "/data/nemotron_data/pretraining_sft_text_document",
        0.131,
        "/data/nemotron_data/high_quality_text_document",
        0.062,
        "/data/nemotron_data/diverse_qa_text_document",
        0.080,
        "/data/hbs_data/finepdfs_bos_text_document",
        0.119,
        "/data/hbs_data/finepdfs_hrv_text_document",
        0.103,
        "/data/hbs_data/finepdfs_srp_text_document",
        0.098,
        "/data/slovene_data/finepdfs_slv_text_document",
        0.030,
        "/data/slovene_data/trendi_text_document",
        0.112,
        "/data/slovene_data/kas_extension_text_document",
        0.072,
        "/data/slovene_data/math_sl_text_document",
        0.077,
        "/data/slovene_data/nemotron_sft_translated_text_document",
    ]
    split = "9900,97,3"
    index_mapping_dir = "/data/12b_train_idx"

    param_dict = {
        "paths": paths,
        "split": split,
        "index_mapping_dir": index_mapping_dir
    }

    return param_dict