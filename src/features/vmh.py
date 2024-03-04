from src.config import VMH_RAW_DATASET_DIRECTOR
from src.utils import json_load
import os


def _get_table(table_name):
    return json_load(os.path.join(VMH_RAW_DATASET_DIRECTOR, f"{table_name}.json"))


def get_raw_metabolites():
    met_data = _get_table("metabolites")
    return met_data


def get_raw_reactions():
    return _get_table("reactions")


def get_raw_metabolite_to_reaction_matrix():
    return _get_table("smatrix")


def get_raw_genes():
    return _get_table("genes")


def get_raw_recons():
    return _get_table("reconstructions")


def get_raw_reaction_to_recon_matrix():
    return _get_table("rxntomodel")


def get_raw_diseases():
    return _get_table("diseases")


def get_raw_biomarkers():
    return _get_table("biomarkers")


def get_raw_evidence():
    return _get_table("evidence")


def get_raw_microbes():
    return _get_table("microbes")


def get_raw_delta_gs_of_metabolites():
    return _get_table("metdeltags")


def get_raw_delta_gs_of_reactions():
    return _get_table("reacdeltags")


def get_raw_fermentation_carbon_of_reactions():
    return _get_table("fermcarbon")


def get_raw_compartments():
    return _get_table("compartments")


def get_raw_foods():
    return _get_table("foods")


def get_raw_subsystem_reactions():
    return _get_table("subsystems")


def get_raw_diet_list():
    return _get_table("diets")


def get_raw_diet_flux_list():
    return _get_table("dietflux")


def get_raw_mass_of_metabolites():
    return _get_table("mmass")


def get_raw_statistics_of_reconstructions():
    return _get_table("numchar")


def get_raw_gene_to_reconstruction():
    return _get_table("genetomodel")


def get_raw_food_groups():
    return _get_table("foodgroups")


def get_raw_nutrients():
    return _get_table("nutrients")


def get_raw_nutrients_to_food():
    return _get_table("nutritiondata")


def get_raw_microbe_genes():
    return _get_table("microbegenes")


def get_raw_microbe_genes_to_recons():
    return _get_table("reactiontomicrobegene")


def get_raw_body_loc_of_microbes():
    return _get_table("bodylocations")


def get_raw_comparative_genomic_analysis():
    return _get_table("compgenstatus")
