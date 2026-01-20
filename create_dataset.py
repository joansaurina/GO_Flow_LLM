import polars as pl
import re
import requests
from functools import lru_cache
from pathlib import Path
from epmc_xml import fetch
from ratelimit.exception import RateLimitException
import time


def is_open_access(pmcid):
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

    paper_url = url.format(pmcid=pmcid)
    r = requests.get(paper_url)
    return r.status_code == 200


@lru_cache
def _get_article(pmcid):
    try:
        art = fetch.article(pmcid)
        time.sleep(0.1)
    except RateLimitException:
        print("Ratelimit exceeded, having a 5 second nap")
        time.sleep(5)
        art = fetch.article(pmcid)

    return art


def identify_used_ids(args):
    pmcid = args["PMCID"]
    genes = args["Gene Names"]
    rnas = args["rna_id"]

    article = _get_article(pmcid)
    full_text = "\n\n".join(list(article.get_sections().values()))
    sentences = full_text.split(".")

    used_rna_id = None
    used_prot_id = None
    if len(rnas) == 1:
        ## Then the URS was unresolved, we should pass
        ## it back as-is for later manual fixing
        used_rna_id = rnas[0]
    else:
        # Find the most mentioned RNA ID:
        rna_mentions = {rna: 0 for rna in rnas}
        for sentence in sentences:
            for rna in rnas:
                r = re.search(f".*{rna}.*", sentence, re.IGNORECASE)
                if r is not None:
                    rna_mentions[rna] += 1

    if genes[0] is None:
        used_prot_id = "N/A"
    else:
        # Find the most mentioned protein:
        prot_mentions = {prot: 0 for prot in genes}
        for sentence in sentences:
            for prot in genes:
                r = re.search(f".*{prot}.*", sentence, re.IGNORECASE)
                if r is not None:
                    prot_mentions[prot] += 1
                    # print(sentence)
    ## Select the most specific RNA Identifier we can
    ## based on its length

    def select_id(mentions):
        selected_id = None
        for k in sorted(mentions.keys(), key=lambda x: len(x), reverse=True):
            ## Selects the longest key that has nonzero mentions
            if mentions[k] > 0:
                selected_id = k
                break
        ## Returns none if none of the ids was found
        return selected_id

    if used_rna_id is None:
        used_rna_id = select_id(rna_mentions)
    if used_prot_id is None:
        used_prot_id = select_id(prot_mentions)

    return {"used_protein_id": used_prot_id, "used_rna_id": used_rna_id}


def expand_extension(ext):
    if ext is None or ext == "":
        return {"targets": list(), "anatomical_locations": list(), "cell_lines": list()}

    def get_input(ext_text):
        protein = re.match(r".*has_input\(UniProtKB:([A-Za-z0-9]+)\)", ext_text)
        if protein:
            protein = protein.group(1)
            return protein
        return None

    def get_anatomy(ext_text):
        location = re.match(r".*occurs_in\(UBERON:([0-9]+)\)", ext_text)
        if location:
            location = location.group(1)
            return f"UBERON:{location}"

        return None

    def get_cell_line(ext_text):
        location = re.match(r".*occurs_in\(CL:([0-9]+)\)", ext_text)
        if location:
            location = location.group(1)
            return f"CL:{location}"

        return None

    proteins = []
    anatomies = []
    cell_lines = []
    for sub_ext in ext.split("|"):
        protein = get_input(sub_ext)
        anatomy = get_anatomy(sub_ext)
        cell_line = get_cell_line(sub_ext)
        proteins.append(protein)
        anatomies.append(anatomy)
        cell_lines.append(cell_line)

    return {
        "targets": list(set(proteins)),
        "anatomical_locations": list(set(anatomies)),
        "cell_lines": list(set(cell_lines)),
    }


def assign_classes(df):
    """
    Loop over the dataframe, look at what is known about a paper's annotations and make a classification on that basis
    """
    # 1. Identify papers that have the mechanism annotation (GO:1903231 with qualifier 'enables')
    # We group by pmid and check if the condition exists in the group
    mechanism_papers = (
        df.filter(
            (pl.col("qualifier") == "enables") & (pl.col("go_term") == "GO:1903231")
        )
        .select("pmid")
        .unique()
        .with_columns(pl.lit(True).alias("has_mechanism"))
    )

    # 2. Join back to df
    df_with_mech = df.join(mechanism_papers, on="pmid", how="left").with_columns(
        pl.col("has_mechanism").fill_null(False)
    )

    # 3. Assign classes based on go_term and has_mechanism
    target_terms = ["GO:0035195", "GO:0035278", "GO:0035279"]
    result_df = df_with_mech.filter(pl.col("go_term").is_in(target_terms))

    result_df = result_df.with_columns(
        pl.when(pl.col("has_mechanism"))
        .then(
            pl.when(pl.col("go_term") == "GO:0035195").then(1)
            .when(pl.col("go_term") == "GO:0035279").then(2)
            .when(pl.col("go_term") == "GO:0035278").then(3)
            .otherwise(4)
        )
        .otherwise(4)
        .alias("class")
    )

    # 4. Select unique papers (first occurrence) and format columns
    result_df = result_df.unique(subset=["PMCID"], keep="first")

    return result_df.select([
        pl.col("used_protein_id").alias("protein_id"),
        "rna_id",
        "date",
        "class",
        "go_term",
        "PMCID"
    ])


def main():
    ## This is processed out of the goa_rna_all.gpa file from here:
    ## https://ftp.ebi.ac.uk/pub/contrib/goa/goa_rna_all.gpa.gz
    raw = pl.read_csv(
        "data/bhf_ucl_annotations.tsv",
        separator="\t",
        has_header=False,
        columns=[1, 2, 3, 4, 8, 10],
        new_columns=["rna_id", "qualifier", "go_term", "pmid", "date", "extension"],
        infer_schema_length=None,
        dtypes={
            "rna_id": pl.Utf8,
            "qualifier": pl.Utf8,
            "go_term": pl.Utf8,
            "pmid": pl.Utf8,
            "extension": pl.Utf8,
        },
    )
    raw = raw.with_columns(pl.col("pmid").str.split(":").list.last())
    raw = raw.with_columns(
        res=pl.col("extension").map_elements(
            expand_extension,
            return_dtype=pl.Struct(
                [
                    pl.Field("targets", pl.List(pl.Utf8)),
                    pl.Field("anatomical_locations", pl.List(pl.Utf8)),
                    pl.Field("cell_lines", pl.List(pl.Utf8)),
                ]
            ),
        )
    ).unnest("res")

    ## Downloaded from https://ftp.ncbi.nlm.nih.gov/pub/pmc/PMC-ids.csv.gz
    pmid_pmcid_mapping = pl.scan_csv(
        "data/PMID_PMCID_DOI.csv",
    )

    raw = (
        raw.lazy()
        .join(pmid_pmcid_mapping, left_on="pmid", right_on="PMID")
        .filter(pl.col("PMCID").is_not_null())
        .collect()
    )

    ## Select unique papers
    ## Explode list of targets
    ## Filter for only entries with a target (should be our terms)
    targets = raw.unique("pmid").explode("targets").filter(pl.col("targets").is_not_null())
    print(f"Total unique papers: {targets.height}")
    cached_targets = False  # True
    if cached_targets and Path("data/bhf_cached_target_data.parquet").exists():
        targets = pl.read_parquet("data/bhf_cached_target_data.parquet")
    else:
        uniprot_ids = pl.read_csv("data/idmapping_uniprot.tsv", separator="\t")
        targets = targets.join(uniprot_ids, left_on="targets", right_on="Entry")
        targets = targets.with_columns(pl.col("Gene Names").str.split(" ")).explode(
            "Gene Names"
        )
        ## Expand gpa data to get PMCIDs - so we can check OA status
        targets = (
            targets.lazy()
            .join(pmid_pmcid_mapping, left_on="pmid", right_on="PMID")
            .filter(pl.col("PMCID").is_not_null())
            .collect()
        )
        ## Use ePMC API to check if we can pull the xml
        targets = targets.with_columns(
            open_access=pl.col("PMCID").map_elements(
                is_open_access, return_dtype=pl.Boolean
            )
        ).filter(pl.col("open_access"))
        print(f"Number of open acces pblicatins available: {targets.height}")

        # Load RNAcentral mapping once instead of per-row
        rnacentral_mapping = pl.scan_csv(
            "data/id_mapping.tsv",
            separator="\t",
            has_header=False,
            new_columns=["urs", "source", "external_id", "taxid", "type", "synonym"],
        ).filter(pl.col("source") == "MIRBASE").collect()

        # Prepare targets for join
        targets = targets.with_columns(
            pl.col("rna_id").str.split("_").list.get(0).alias("urs"),
            pl.col("rna_id").str.split("_").list.get(1).cast(pl.Int64).alias("taxid")
        )

        # Join with mapping
        targets = targets.join(rnacentral_mapping, on=["urs", "taxid"], how="left")

        # Construct the ID string efficiently
        targets = targets.with_columns(
            pl.when(pl.col("external_id").is_not_null())
            .then(
                pl.format("{}|{}|{}", 
                    pl.col("external_id"), 
                    pl.col("synonym"), 
                    pl.col("synonym").str.split("-").list.slice(1, 2).list.join("-")
                )
            )
            .otherwise(pl.col("rna_id"))
            .alias("rna_id_mapped")
        ).drop(["urs", "taxid", "source", "external_id", "type", "synonym", "rna_id"]).rename({"rna_id_mapped": "rna_id"})

        targets.write_parquet("data/bhf_cached_target_data.parquet")

    targets = targets.with_columns(pl.col("rna_id").str.split("|")).explode("rna_id")

    ## paper and targets is the manually checked rna, and protein ISd for each paper
    if not Path("data/paper_and_targets.csv").exists():
        paper_searching = (
            targets.group_by("PMCID")
            .agg(pl.col("Gene Names").unique(), pl.col("rna_id").unique())
            .sort(by="PMCID")
        )
        paper_searching = paper_searching.with_columns(
            res=pl.struct("PMCID", "Gene Names", "rna_id").map_elements(
                identify_used_ids, return_dtype=pl.Struct
            )
        )
        paper_searching = paper_searching.unnest("res")
        paper_searching.select(["PMCID", "used_protein_id", "used_rna_id"]).write_csv(
            "data/paper_and_targets.csv"
        )
        # After writing, manually check and fix anything missing
    else:
        paper_searching = pl.read_csv("data/paper_and_targets.csv")

    enriched_target_data = raw.select(
        ["pmid", "PMCID", "go_term", "date", "extension", "qualifier"]
    ).join(paper_searching, on="PMCID", how="inner")

    classification_data = assign_classes(enriched_target_data)
    classification_data.write_parquet("data/bhf_paper_classification_data.parquet")
    print(classification_data)


if __name__ == "__main__":
    main()
