import argparse
import faulthandler
import json
import logging
import os
import sys
import polars as pl

from pathlib import Path
from epmc_xml import fetch
from pydantic import ValidationError
from guidance import system, user

from mirna_curator.flowchart import curation, flow_prompts
from mirna_curator.flowchart.computation_graph import ComputationGraph
from mirna_curator.model.llm import get_model
from mirna_curator.utils.tracing import curation_tracer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def arg_parser():
    parser = argparse.ArgumentParser(description="GoFlowLLM miRNA Curator")
    parser.add_argument("--config", type=str, help="Path to a config.json file with options for the run set")
    parser.add_argument("--model_path", type=str, help="A huggingface ID or local model path")
    parser.add_argument("--flowchart", type=str, help="The flowchart, defined in JSON")
    parser.add_argument("--prompts", type=str, help="The prompts, defined in JSON")
    parser.add_argument("--context_length", type=int, default=16384, help="The context length for the model")
    parser.add_argument("--quantization", type=str, help="The quantization for the model")
    parser.add_argument("--chat_template", type=str, help="The chat template for the model")
    parser.add_argument("--input_data", type=str, help="The input data (PMCID and detected RNA ID) for the process")
    parser.add_argument("--output_data", type=str, help="The output data (curation result) for the process")
    parser.add_argument("--max_papers", type=int, help="The maximum number of papers to process")
    parser.add_argument("--annot_class", type=int, help="Restrict processing to one class of annotation")
    parser.add_argument("--validate_only", action="store_true", help="only load and validate flowcharts")
    parser.add_argument("--evidence_type", type=str, default="single-sentence", choices=["recursive-paragraph", "recursive-sentence", "single-sentence", "single-paragraph", "full-substring"], help="How to do the evidence extraction")
    parser.add_argument("--deepseek_mode", action="store_true", help="Tweak the reasoning generation for deepseek models")
    parser.add_argument("--checkpoint_frequency", type=int, default=-1, help="How often to write a results checkpoint")
    parser.add_argument("--checkpoint_file_path", type=str, default="curation_results_checkpoint.parquet", help="Name of the file to checkpoint into")
    parser.add_argument("--gpu", type=str, default="0", help="Which gpu ID to run on, if there are several available")

    args = parser.parse_args()

    if args.config:
        try:
            with open(args.config, "r") as f:
                config = json.load(f)
                vars(args).update(config)
        except Exception as e:
            logger.fatal(f"Error reading config file: {e}")
            sys.exit(1)
    return args

def main():
    # Load files:
    args = arg_parser()
    curation_tracer.set_model_name(args.model_path)

    run_config_options = {"evidence_mode": args.evidence_type, "deepseek_mode": args.deepseek_mode}
    try:
        cur_flowchart_string = open(args.flowchart, "r").read()
        cf = curation.CurationFlowchart.model_validate_json(cur_flowchart_string)
    except ValidationError as e:
        logger.fatal(e)
        logger.fatal("Error loading flowchart, aborting")
        sys.exit(1)
    logger.info(f"Loaded flowchart from {args.flowchart}")

    try:
        prompt_string = open(args.prompts, "r").read()
        prompt_data = flow_prompts.CurationPrompts.model_validate_json(prompt_string)
    except ValidationError as e:
        logger.fatal(e)
        logger.fatal("Error loading prompts, aborting")
        sys.exit(1)
    logger.info(f"Loaded prompts from {args.prompts}")

    if args.validate_only:
        logger.info("Validation only, exiting now")
        sys.exit(0)

    # Validate arguments:
    if any([args.model_path is None, args.flowchart is None, args.prompts is None,
        args.input_data is None, args.output_data is None, args.chat_template is None]):
        logger.error("A required argument is set to None, check your config!")
        sys.exit(1)

    # Set GPU and Model:
    if args.gpu is not None:
        logger.info("Selecting %s gpu for this process", args.gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    llm = get_model(args.model_path, chat_template=args.chat_template, quantization=args.quantization,
                context_length=args.context_length)
    logger.info(f"Loaded model from {args.model_path}")

    graph = ComputationGraph(cf, run_config=run_config_options)
    logger.info("Constructed computation graph")

    # Get the curation input data 
    if args.input_data.endswith("parquet") or args.input_data.endswith("pq"):
        curation_input = pl.read_parquet(args.input_data)
    elif args.input_data.endswith("csv"):
        curation_input = pl.read_csv(args.input_data)
    else:
        logger.error("Unsupported input data format for %s", args.input_data)
        sys.exit(1)

    # Resume if there's a valid checkpoint:
    if Path(args.checkpoint_file_path).exists():
        logger.info("Resuming from checkpoint %s", args.checkpoint_file_path)
        done = pl.read_parquet(args.checkpoint_file_path)
        curation_input = curation_input.join(done, on="PMCID", how="anti")

    # UNUSED for now dont't delete:
    for prompt in prompt_data.prompts:
        if prompt.type == "system":
            logger.info("Found system prompt, applying...")
            try:
                with system():
                    llm += prompt.prompt
            except Exception as e:
                logger.warning("Selected model does not have a system prompt mode, forward as user instead")
                with user():
                    llm += prompt.prompt
            break
    if args.annot_class is not None:
        logger.info(f"Restricting processing to annotation class {args.annot_class}")
        curation_input = curation_input.filter(pl.col("class") == args.annot_class)

    logger.info(f"Loaded input data from {args.input_data}")
    logger.info(f"Processing up to {curation_input.height} papers")

    curation_output = []
    
    # Loop:
    for i, row in enumerate(curation_input.iter_rows(named=True)):
        if args.max_papers is not None and i >= args.max_papers:
            break
        if args.checkpoint_frequency > 0 and i > 0 and i % args.checkpoint_frequency == 0:
            logger.info("Checkpointing results")
            if len(curation_output) > 0:
                curation_output_df = pl.DataFrame(curation_output)
                if Path(args.checkpoint_file_path).exists():
                    prev = pl.read_parquet(args.checkpoint_file_path)
                    pl.concat([curation_output_df, prev], how="diagonal_relaxed")
                else:
                    curation_output_df.write_parquet(args.checkpoint_file_path)
        try:
            logger.info("Starting curation for paper %s", row["PMCID"])
            article = fetch.article(row["PMCID"])
            article.add_figures_section()
        except Exception as e:
            logger.error(e)
            logger.error(f"Failed to fetch/parse {row['PMCID']}, skipping it")
            continue
        try:
            llm_trace, curation_result = graph.execute_graph(row["PMCID"], llm, article, row["rna_id"], prompt_data)
        except Exception as e:
            logger.error(e)
            logger.error("Paper %s has exceeded context limit, skipping", row["PMCID"])
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
            continue
        logger.info(f"RNA ID: {row['rna_id']} in {row['PMCID']} - Curation Result: {curation_result}")
        curation_output.append({"PMCID": row["PMCID"], "rna_id": row["rna_id"], "curation_result": curation_result})
    
    curation_output_df = pl.DataFrame(curation_output)
    curation_output_df.write_parquet(args.output_data)

if __name__ == "__main__":
    main()
