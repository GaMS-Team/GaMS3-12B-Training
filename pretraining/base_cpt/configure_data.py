def get_data_parameters():
    paths = [
        0.019,
        "/data/nemotron_data/nemotron_pretraining_code_text_document",
        0.025,
        "/data/nemotron_data/nemotron_math_4_plus_text_document",
        0.012,
        "/data/nemotron_data/nemotron_math_3_text_document",
        0.037,
        "/data/nemotron_data/nemotron_pretraining_sft_text_document",
        0.104,
        "/data/nemotron_data/nemotron_high_quality_text_document",
        0.086,
        "/data/nemotron_data/nemotron_diverse_qa_text_document",
        0.048,
        "/data/hbs_data/finepdfs_bos_text_document",
        0.094,
        "/data/hbs_data/finepdfs_hrv_text_document",
        0.080,
        "/data/hbs_data/finepdfs_srp_text_document",
        0.059,
        "/data/slovene_data/finepdfs_slv_text_document",
        0.017,
        "/data/slovene_data/trendi_text_document",
        0.042,
        "/data/slovene_data/classla_text_document",
        0.017,
        "/data/slovene_data/sl_legal_text_document",
        0.016,
        "/data/slovene_data/sl_med_text_document",
        0.045,
        "/data/slovene_data/metafida_text_document",
        0.138,
        "/data/slovene_data/fineweb2_text_document",
        0.027,
        "/data/slovene_data/kas_text_document",
        0.012,
        "/data/slovene_data/nuk_combined_text_document",
        0.115,
        "/data/slovene_data/nuk_doc_text_document",
        0.007,
        "/data/wikipedia/wikipedia_yugo_text_document"
    ]
    split = "9900,99,1"
    index_mapping_dir = "/data/12b_train_idx"

    param_dict = {
        "paths": paths,
        "split": split,
        "index_mapping_dir": index_mapping_dir
    }

    return param_dict