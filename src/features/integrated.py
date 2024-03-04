import dataclasses
from dataclasses import dataclass

from src.config import INTEGRATED_DATA_DIR
from src.utils.io import json_load, json_dump

from src.features.vmh import get_raw_subsystem_reactions, get_raw_reactions, get_raw_microbes, \
    get_raw_body_loc_of_microbes, get_raw_metabolites, get_raw_diseases, get_raw_compartments, \
    get_raw_reaction_to_recon_matrix, get_raw_metabolite_to_reaction_matrix, get_raw_biomarkers
import pandas as pd

ENTITY_TYPES = ['Entity',
                'Microbe',
                'Subsystem',
                'Reaction',
                'Metabolite',
                'Disease',
                'BodyLocation',
                'Compartment',
                ]

RELATION_TYPES = ['Relation',
                  'MicrobeLocated',
                  'LocHasMicrobe',
                  'MicrobeHasReact',
                  'ReactInSub',
                  'SubHasReact',
                  'ReactantIn',
                  'ProductIn',
                  'HasProduct',
                  'HasReactant',
                  'Biomarker',
                  'CompHasMet',
                  'MetInComp',
                  'MicrobeCauseDisease',
                  'DiseaseRelateToMicrobe'
                  ]


def generate_file_name(name):
    return f'{INTEGRATED_DATA_DIR}/{name}.json'


ENTITY_FILE_NAME = generate_file_name(ENTITY_TYPES[0])
MICROBE_FILE_NAME = generate_file_name(ENTITY_TYPES[1])
SUBSYSTEM_FILE_NAME = generate_file_name(ENTITY_TYPES[2])
REACTION_FILE_NAME = generate_file_name(ENTITY_TYPES[3])
METABOLITE_FILE_NAME = generate_file_name(ENTITY_TYPES[4])
DISEASE_FILE_NAME = generate_file_name(ENTITY_TYPES[5])
BODY_LOCATION_FILE_NAME = generate_file_name(ENTITY_TYPES[6])
COMPARTMENT_FILE_NAME = generate_file_name(ENTITY_TYPES[7])

RELATION_FILE_NAME = generate_file_name(RELATION_TYPES[0])
MICROBE_LOCATED_FILE_NAME = generate_file_name(RELATION_TYPES[1])
LOC_HAS_MICROBE_FILE_NAME = generate_file_name(RELATION_TYPES[2])
MICROBE_HAS_REACT_FILE_NAME = generate_file_name(RELATION_TYPES[3])
REACT_IN_SUB_FILE_NAME = generate_file_name(RELATION_TYPES[4])
SUB_HAS_REACT_FILE_NAME = generate_file_name(RELATION_TYPES[5])
REACTANT_IN_FILE_NAME = generate_file_name(RELATION_TYPES[6])
PRODUCT_IN_FILE_NAME = generate_file_name(RELATION_TYPES[7])
HAS_PRODUCT_FILE_NAME = generate_file_name(RELATION_TYPES[8])
HAS_REACTANT_FILE_NAME = generate_file_name(RELATION_TYPES[9])
BIOMARKER_FILE_NAME = generate_file_name(RELATION_TYPES[10])
COMP_HAS_MET_FILE_NAME = generate_file_name(RELATION_TYPES[11])
MET_IN_COMP_FILE_NAME = generate_file_name(RELATION_TYPES[12])
MICROBE_CAUSE_DISEASE_FILE_NAME = generate_file_name(RELATION_TYPES[13])
DISEASE_RELATE_TO_MICROBE_FILE_NAME = generate_file_name(RELATION_TYPES[14])


@dataclass
class Entity:
    id: int


@dataclass
class MicrobeEntity(Entity):
    organism: str
    kingdom: str
    phylum: str
    mclass: str
    order: str
    family: str
    genus: str
    oxygenstat: str
    metabolism: str
    gram: str
    mtype: str


@dataclass
class SubsystemEntity(Entity):
    name: str


@dataclass
class BodyLocationEntity(Entity):
    name: str


@dataclass
class CompartmentEntity(Entity):
    name: str


@dataclass
class ReactionEntity(Entity):
    formula: str
    reversible: bool
    isHuman: bool
    isMicrobe: bool
    abbreviation: str


@dataclass
class MetaboliteEntity(Entity):
    abbreviation: str
    fullName: str
    chargedFormula: str
    charge: int
    avgmolweight: float
    monoisotopicweight: float
    smile: str
    isHuman: bool
    isMicrobe: bool


@dataclass
class DiseaseEntity(Entity):
    abbreviation: str
    name: str
    subtype: str
    inheritance: str
    organ: str


@dataclass
class Relation:
    id: int
    head_id: int
    tail_id: int


@dataclass
class MicrobeLocatedRelation(Relation):
    pass


@dataclass
class LocHasMicrobeRelation(Relation):
    pass


@dataclass
class MicrobeHasReactRelation(Relation):
    lb: float
    ub: float


@dataclass
class ReactInSubRelation(Relation):
    pass


@dataclass
class SubHasReactRelation(Relation):
    pass


@dataclass
class MetaboliteInRelation(Relation):
    value: float


@dataclass
class HasMetaboliteRelation(Relation):
    value: float


@dataclass
class HasReactantRelation(Relation):
    value: float


@dataclass
class BiomarkerRelation(Relation):
    value: str
    normalConcentrationUp: float
    normalConcentrationDown: float
    rangeConcentrationUp: float
    rangeConcentrationDown: float


@dataclass
class CompHasMetRelation(Relation):
    pass


@dataclass
class MetInCompRelation(Relation):
    pass


def _get_table(table_name):
    return json_load(table_name)


def get_entities():
    return _get_table(ENTITY_FILE_NAME)


def get_body_locations():
    return _get_table(BODY_LOCATION_FILE_NAME)


def get_microbes():
    return _get_table(MICROBE_FILE_NAME)


def get_subsystems():
    return _get_table(SUBSYSTEM_FILE_NAME)


def get_reactions():
    return _get_table(REACTION_FILE_NAME)


def get_metabolites():
    return _get_table(METABOLITE_FILE_NAME)


def get_diseases():
    return _get_table(DISEASE_FILE_NAME)


def get_compartments():
    return _get_table(COMPARTMENT_FILE_NAME)


def get_relations():
    return _get_table(RELATION_FILE_NAME)


def get_microbe_located_relations():
    return _get_table(MICROBE_LOCATED_FILE_NAME)


def get_loc_has_microbe_relations():
    return _get_table(LOC_HAS_MICROBE_FILE_NAME)


def get_microbe_has_react_relations():
    return _get_table(MICROBE_HAS_REACT_FILE_NAME)


def get_react_in_sub_relations():
    return _get_table(REACT_IN_SUB_FILE_NAME)


def get_sub_has_react_relations():
    return _get_table(SUB_HAS_REACT_FILE_NAME)


def get_reactant_in_relations():
    return _get_table(REACTANT_IN_FILE_NAME)


def get_product_in_relations():
    return _get_table(PRODUCT_IN_FILE_NAME)


def get_has_reactant_relations():
    return _get_table(HAS_REACTANT_FILE_NAME)


def get_has_product_relations():
    return _get_table(HAS_PRODUCT_FILE_NAME)


def get_biomarker_relations():
    return _get_table(BIOMARKER_FILE_NAME)


def get_comp_has_met_relations():
    return _get_table(COMP_HAS_MET_FILE_NAME)


def get_met_in_comp_relations():
    return _get_table(MET_IN_COMP_FILE_NAME)


def clean_entity_files():
    json_dump(ENTITY_FILE_NAME, list())
    json_dump(BODY_LOCATION_FILE_NAME, list())
    json_dump(MICROBE_FILE_NAME, list())
    json_dump(SUBSYSTEM_FILE_NAME, list())
    json_dump(REACTION_FILE_NAME, list())
    json_dump(METABOLITE_FILE_NAME, list())
    json_dump(DISEASE_FILE_NAME, list())
    json_dump(COMPARTMENT_FILE_NAME, list())


def load_vmh_entities():
    load_vmh_body_location()
    load_vmh_microbes()
    load_vmh_subsystems()
    load_vmh_reaction()
    load_vmh_metabolites()
    load_vmh_diseases()
    load_vmh_compartments()


def clean_relation_files():
    json_dump(RELATION_FILE_NAME, list())
    json_dump(MICROBE_LOCATED_FILE_NAME, list())
    json_dump(LOC_HAS_MICROBE_FILE_NAME, list())
    json_dump(MICROBE_HAS_REACT_FILE_NAME, list())
    json_dump(REACT_IN_SUB_FILE_NAME, list())
    json_dump(SUB_HAS_REACT_FILE_NAME, list())
    json_dump(REACTANT_IN_FILE_NAME, list())
    json_dump(PRODUCT_IN_FILE_NAME, list())
    json_dump(HAS_REACTANT_FILE_NAME, list())
    json_dump(HAS_PRODUCT_FILE_NAME, list())
    json_dump(BIOMARKER_FILE_NAME, list())
    json_dump(COMP_HAS_MET_FILE_NAME, list())
    json_dump(MET_IN_COMP_FILE_NAME, list())


def load_vmh_relations():
    load_vmh_microbe_located_relations()
    load_vmh_loc_has_microbe_relations()
    load_vmh_microbe_has_react_relations()
    load_vmh_react_in_sub_relations()
    load_vmh_sub_has_react_relations()
    load_vmh_reactant_in_relations()
    load_vmh_product_in_relations()
    load_vmh_has_product_relations()
    load_vmh_has_reactant_relations()
    load_vmh_biomarker_relations()
    load_vmh_comp_has_met_relations()
    load_vmh_met_in_comp_relations()


_cached_entities = []
_cached_relations = []


def _dump_entities():
    global _cached_entities
    json_dump(ENTITY_FILE_NAME, _cached_entities)
    _cached_entities = []


def _dump_relations():
    global _cached_relations
    json_dump(RELATION_FILE_NAME, _cached_relations)
    _cached_relations = []


def _add_entity(t: ENTITY_TYPES, save=False):
    global _cached_entities
    if len(_cached_entities) == 0:
        _cached_entities = json_load(ENTITY_FILE_NAME) if json_load(ENTITY_FILE_NAME) is not None else list()

    new_id = _get_last_entity_id() + 1
    _cached_entities.append({'id': new_id, 'type': t})
    if save:
        _dump_entities()
    return new_id


def _add_relation(t: RELATION_TYPES, head_id: int, tail_id: int, save=False):
    global _cached_relations
    if len(_cached_relations) == 0:
        _cached_relations = json_load(RELATION_FILE_NAME) if json_load(RELATION_FILE_NAME) is not None else list()

    new_id = _get_last_relation_id() + 1
    r = dataclasses.asdict(Relation(id=new_id, head_id=head_id, tail_id=tail_id))
    r['type'] = t
    _cached_relations.append(r)
    if save:
        _dump_relations()
    return new_id


def _get_last_entity_id():
    global _cached_entities

    if len(_cached_entities) != 0:
        return _cached_entities[-1]['id']
    else:
        return 0


def _get_last_relation_id():
    global _cached_relations

    if len(_cached_relations) != 0:
        return _cached_relations[-1]['id']
    else:
        return 0


def _update_file_list(data: list, file_name):
    pre = json_load(file_name) if json_load(file_name) is not None else list()
    new = pre + data
    json_dump(file_name, new)


def _update_a_entity_list(data: list, file_name):
    _update_file_list(data, file_name)
    _dump_entities()


def _update_a_relation_list(data: list, file_name):
    _update_file_list(data, file_name)
    _dump_relations()


def _extract_vmh_subsystems():
    systems = []
    for name in list(pd.DataFrame(get_raw_subsystem_reactions())['name']):
        system_id = _add_entity(ENTITY_TYPES[2])
        systems.append(dataclasses.asdict(SubsystemEntity(system_id, name)))
    return systems


def load_vmh_subsystems():
    _update_a_entity_list(_extract_vmh_subsystems(), SUBSYSTEM_FILE_NAME)


def _extract_vmh_reactions():
    reactions = []
    table = pd.DataFrame(get_raw_reactions())[['formula', 'reversible', 'isHuman', 'isMicrobe', 'abbreviation']]
    for i in range(table.shape[0]):
        rid = _add_entity(ENTITY_TYPES[3])
        reactions.append(dataclasses.asdict(ReactionEntity(id=rid,
                                                           formula=table.iloc[i, 0],
                                                           reversible=bool(table.iloc[i, 1]),
                                                           isHuman=bool(table.iloc[i, 2]),
                                                           isMicrobe=bool(table.iloc[i, 3]),
                                                           abbreviation=table.iloc[i, 4])))
    return reactions


def load_vmh_reaction():
    _update_a_entity_list(_extract_vmh_reactions(), REACTION_FILE_NAME)


def _extract_vmh_microbes():
    microbes = []
    table = pd.DataFrame(get_raw_microbes())[
        ['organism', 'kingdom', 'phylum', 'mclass', 'order', 'family', 'genus', 'oxygenstat', 'metabolism', 'gram',
         'mtype']]

    for i in range(table.shape[0]):
        mid = _add_entity(ENTITY_TYPES[1])
        microbes.append(dataclasses.asdict(MicrobeEntity(id=mid,
                                                         organism=table.iloc[i, 0],
                                                         kingdom=table.iloc[i, 1],
                                                         phylum=table.iloc[i, 2],
                                                         mclass=table.iloc[i, 3],
                                                         order=table.iloc[i, 4],
                                                         family=table.iloc[i, 5],
                                                         genus=table.iloc[i, 6],
                                                         oxygenstat=table.iloc[i, 7],
                                                         metabolism=table.iloc[i, 8],
                                                         gram=table.iloc[i, 9],
                                                         mtype=table.iloc[i, 10], )))
    return microbes


def load_vmh_microbes():
    _update_a_entity_list(_extract_vmh_microbes(), MICROBE_FILE_NAME)


def _extract_vmh_body_locations():
    body_locations = []
    table = pd.DataFrame(get_raw_body_loc_of_microbes())[['bodyLocation']].drop_duplicates()

    for i in range(table.shape[0]):
        bid = _add_entity(ENTITY_TYPES[6])
        body_locations.append(dataclasses.asdict(BodyLocationEntity(id=bid, name=table.iloc[i, 0])))
    return body_locations


def load_vmh_body_location():
    _update_a_entity_list(_extract_vmh_body_locations(), BODY_LOCATION_FILE_NAME)


def _extract_vmh_metabolites():
    metabolites = []
    table = pd.DataFrame(get_raw_metabolites())[[
        'abbreviation',
        'fullName',
        'chargedFormula',
        'charge',
        'avgmolweight',
        'monoisotopicweight',
        'smile',
        'isHuman',
        'isMicrobe',
    ]]

    for i in range(table.shape[0]):
        mid = _add_entity(ENTITY_TYPES[4])
        metabolites.append(dataclasses.asdict(MetaboliteEntity(
            id=mid,
            abbreviation=table.iloc[i, 0],
            fullName=table.iloc[i, 1],
            chargedFormula=table.iloc[i, 2],
            charge=int(table.iloc[i, 3]),
            avgmolweight=float(table.iloc[i, 4]),
            monoisotopicweight=float(table.iloc[i, 5]),
            smile=table.iloc[i, 6],
            isHuman=bool(table.iloc[i, 7]),
            isMicrobe=bool(table.iloc[i, 8])
        )))
    return metabolites


def load_vmh_metabolites():
    _update_a_entity_list(_extract_vmh_metabolites(), METABOLITE_FILE_NAME)


def _extract_vmh_diseases():
    diseases = []
    table = pd.DataFrame(get_raw_diseases())[[
        'abbreviation',
        'name',
        'subtype',
        'inheritance',
        'organ',
    ]]

    for i in range(table.shape[0]):
        did = _add_entity(ENTITY_TYPES[5])
        diseases.append(dataclasses.asdict(DiseaseEntity(
            id=did,
            abbreviation=table.iloc[i, 0],
            name=table.iloc[i, 1],
            subtype=table.iloc[i, 2],
            inheritance=table.iloc[i, 3],
            organ=table.iloc[i, 4],
        )))
    return diseases


def load_vmh_diseases():
    _update_a_entity_list(_extract_vmh_diseases(), DISEASE_FILE_NAME)


def _extract_vmh_compartments():
    compartments = []
    table = pd.DataFrame(get_raw_compartments())[[
        'name',
    ]]

    for i in range(table.shape[0]):
        did = _add_entity(ENTITY_TYPES[7])
        compartments.append(dataclasses.asdict(CompartmentEntity(
            id=did,
            name=table.iloc[i, 0]
        )))
    return compartments


def load_vmh_compartments():
    _update_a_entity_list(_extract_vmh_compartments(), COMPARTMENT_FILE_NAME)


def _extract_vmh_microbe_location_pairs():
    microbe_organism = [row['microbe']['organism'] for row in get_raw_body_loc_of_microbes()]
    body_locations = [row['bodyLocation'] for row in get_raw_body_loc_of_microbes()]

    for o in microbe_organism:
        if o == '' or o is None:
            raise Exception('Microbe organism is not specified microbe located relation!')

    for o in body_locations:
        if o == '' or o is None:
            raise Exception('Microbe location is not specified microbe located relation')

    microbe_ids = [microbe['id'] for organism in microbe_organism for microbe in get_microbes() if
                   microbe['organism'] == organism]
    location_ids = [location['id'] for location_name in body_locations for location in get_body_locations() if
                    location['name'] == location_name]
    return microbe_ids, location_ids


def _extract_vmh_microbe_located_relations():
    relations = []
    microbe_ids, location_ids = _extract_vmh_microbe_location_pairs()
    for p in zip(location_ids, microbe_ids):
        rid = _add_relation(RELATION_TYPES[1], *p)
        relations.append(dataclasses.asdict(MicrobeLocatedRelation(id=rid, head_id=p[0], tail_id=p[1])))
    return relations


def load_vmh_microbe_located_relations():
    _update_a_relation_list(_extract_vmh_microbe_located_relations(), MICROBE_LOCATED_FILE_NAME)


def _extract_vmh_loc_has_microbe_relations():
    relations = []
    microbe_ids, location_ids = _extract_vmh_microbe_location_pairs()
    for p in zip(microbe_ids, location_ids):
        rid = _add_relation(RELATION_TYPES[2], *p)
        relations.append(dataclasses.asdict(LocHasMicrobeRelation(id=rid, head_id=p[0], tail_id=p[1])))
    return relations


def load_vmh_loc_has_microbe_relations():
    _update_a_relation_list(_extract_vmh_loc_has_microbe_relations(), LOC_HAS_MICROBE_FILE_NAME)


def _extract_vmh_microbe_has_react_relations():
    react_model_table = pd.DataFrame(get_raw_reaction_to_recon_matrix())
    clean_microbes_table = pd.DataFrame(get_microbes())
    reactions_table = pd.DataFrame(get_reactions())

    # Clean React to Model Table
    react_model_table['organism'] = react_model_table.apply(lambda d: d['model']['organism'], axis=1)
    react_model_table = react_model_table[['rxn', 'organism', 'lb', 'ub']]

    # Clean Microbes Table
    clean_microbes_table = clean_microbes_table[['id', 'organism']]
    clean_microbes_table = clean_microbes_table.rename(columns={"id": "microbe_id"})

    # Clean Reaction Table
    reactions_table = reactions_table[['id', 'abbreviation']]
    reactions_table = reactions_table.rename(columns={"id": "reaction_id"})

    # Merge Tables
    merged = react_model_table.join(clean_microbes_table.set_index('organism'), on='organism', validate='m:1')
    merged = merged.join(reactions_table.set_index('abbreviation'), on='rxn', validate='m:1')

    # Drop Nans the rows with reaction of human metabolites
    merged = merged.dropna()

    # Extract Relations
    merged['relations'] = merged.apply(
        lambda d: dataclasses.asdict(
            MicrobeHasReactRelation(
                id=_add_relation(t=RELATION_TYPES[3], head_id=int(d['reaction_id']), tail_id=int(d['microbe_id'])),
                head_id=int(d['reaction_id']),
                tail_id=int(d['microbe_id']), lb=d['lb'],
                ub=d['ub'])
        ),
        axis=1
    )
    relations = merged['relations'].tolist()
    return relations


def load_vmh_microbe_has_react_relations():
    _update_a_relation_list(_extract_vmh_microbe_has_react_relations(), MICROBE_HAS_REACT_FILE_NAME)


def _get_reaction_to_subsystem_pairs():
    react_systems_table = pd.DataFrame(get_raw_subsystem_reactions())
    raw_reactions_table = pd.DataFrame(get_raw_reactions())
    reaction_table = pd.DataFrame(get_reactions())
    subsystem_table = pd.DataFrame(get_subsystems())

    # Build Reaction to Name Table
    react_to_name = []
    for i in range(react_systems_table.shape[0]):
        for rxn in react_systems_table.iloc[i]['rxns']:
            react_to_name.append({'system_name': react_systems_table.iloc[i]['name'],
                                  'reaction_id': rxn})
    react_to_name = pd.DataFrame(react_to_name)

    # Add Reaction Abbreviations
    merged = react_to_name.join(raw_reactions_table.set_index('rxn_id'), on='reaction_id')
    merged = merged[['system_name', 'abbreviation']]

    # Add System ID
    merged = merged.join(subsystem_table.set_index('name'), on='system_name')
    merged = merged.rename(columns={'id': 'subsystem_id'})

    # Add Reaction ID
    merged = merged.join(reaction_table.set_index('abbreviation'), on='abbreviation')
    merged = merged.rename(columns={'id': 'reaction_id'})

    # Clean Table
    merged = merged[['reaction_id', 'subsystem_id']]

    return merged


def _extract_vmh_react_in_sub_relations():
    pairs = _get_reaction_to_subsystem_pairs()
    pairs['relations'] = pairs.apply(
        lambda d: dataclasses.asdict(ReactInSubRelation(
            id=_add_relation(t=RELATION_TYPES[4], head_id=int(d['subsystem_id']), tail_id=int(d['reaction_id'])),
            head_id=int(d['subsystem_id']),
            tail_id=int(d['reaction_id']))
        ),
        axis=1
    )
    relations = pairs['relations'].tolist()
    return relations


def load_vmh_react_in_sub_relations():
    _update_a_relation_list(_extract_vmh_react_in_sub_relations(), REACT_IN_SUB_FILE_NAME)


def _extract_vmh_sub_has_react_relations():
    pairs = _get_reaction_to_subsystem_pairs()
    pairs['relations'] = pairs.apply(
        lambda d: dataclasses.asdict(SubHasReactRelation(
            id=_add_relation(t=RELATION_TYPES[5], head_id=int(d['reaction_id']), tail_id=int(d['subsystem_id'])),
            head_id=int(d['reaction_id']),
            tail_id=int(d['subsystem_id']))
        ),
        axis=1
    )
    relations = pairs['relations'].tolist()
    return relations


def load_vmh_sub_has_react_relations():
    _update_a_relation_list(_extract_vmh_sub_has_react_relations(), SUB_HAS_REACT_FILE_NAME)


def _get_reaction_metabolite_pairs(is_product, is_from_met, t):
    metabolite_to_reaction_table = pd.DataFrame(get_raw_metabolite_to_reaction_matrix())
    metabolite_table = pd.DataFrame(get_metabolites())
    reaction_table = pd.DataFrame(get_reactions())

    # Clean Metabolite Reaction Pairs
    metabolite_to_reaction_table['reaction_abbreviation'] = metabolite_to_reaction_table.apply(
        lambda d: d['rxn']['abbreviation'], axis=1)
    metabolite_to_reaction_table['metabolite_abbreviation'] = metabolite_to_reaction_table.apply(
        lambda d: d['met']['abbreviation'], axis=1)
    metabolite_to_reaction_table = metabolite_to_reaction_table[
        ['reaction_abbreviation', 'metabolite_abbreviation', 'value']]

    # Add Metabolite ID
    merged = metabolite_to_reaction_table.join(metabolite_table.set_index('abbreviation'), on='metabolite_abbreviation')
    merged = merged[['reaction_abbreviation', 'metabolite_abbreviation', 'value', 'id']]
    merged = merged.rename(columns={'id': 'met_id'})

    # Add Reaction ID
    merged = merged.join(reaction_table.set_index('abbreviation'), on='reaction_abbreviation')
    merged = merged[['value', 'met_id', 'id']]
    merged = merged.rename(columns={'id': 'react_id'})

    if is_product:
        merged = merged[merged['value'] >= 0]
    else:
        merged = merged[merged['value'] < 0]

    if is_from_met:
        merged['relations'] = merged.apply(
            lambda d: dataclasses.asdict(
                HasMetaboliteRelation(
                    id=_add_relation(t=t, head_id=int(d['met_id']), tail_id=int(d['react_id'])),
                    head_id=int(d['met_id']),
                    tail_id=int(d['react_id']),
                    value=d['value']
                )
            ),
            axis=1
        )
    else:
        merged['relations'] = merged.apply(
            lambda d: dataclasses.asdict(
                MetaboliteInRelation(
                    id=_add_relation(t=RELATION_TYPES[6], head_id=int(d['react_id']), tail_id=int(d['met_id'])),
                    head_id=int(d['react_id']),
                    tail_id=int(d['met_id']),
                    value=d['value']
                )
            ),
            axis=1
        )

    return merged['relations'].tolist()


def _extract_vmh_reactant_in_relations():
    return _get_reaction_metabolite_pairs(is_product=False, is_from_met=False, t=RELATION_TYPES[6])


def load_vmh_reactant_in_relations():
    _update_a_relation_list(_extract_vmh_reactant_in_relations(), REACTANT_IN_FILE_NAME)


def _extract_vmh_product_in_relations():
    return _get_reaction_metabolite_pairs(is_product=True, is_from_met=False, t=RELATION_TYPES[7])


def load_vmh_product_in_relations():
    _update_a_relation_list(_extract_vmh_product_in_relations(), PRODUCT_IN_FILE_NAME)


def _extract_vmh_has_product_relations():
    return _get_reaction_metabolite_pairs(is_product=True, is_from_met=True, t=RELATION_TYPES[8])


def load_vmh_has_product_relations():
    _update_a_relation_list(_extract_vmh_has_product_relations(), HAS_PRODUCT_FILE_NAME)


def _extract_vmh_has_reactant_relations():
    return _get_reaction_metabolite_pairs(is_product=False, is_from_met=True, t=RELATION_TYPES[9])


def load_vmh_has_reactant_relations():
    _update_a_relation_list(_extract_vmh_has_reactant_relations(), HAS_REACTANT_FILE_NAME)


def _extract_vmh_biomarker_relations():
    biomarker_table = pd.DataFrame(get_raw_biomarkers())
    disease_table = pd.DataFrame(get_diseases())
    metabolite_table = pd.DataFrame(get_metabolites())

    # Clean Biomarker Table
    biomarker_table['met_abbreviation'] = biomarker_table.apply(lambda d: d['metabolite']['abbreviation'], axis=1)
    biomarker_table['disease_abbreviation'] = biomarker_table.apply(lambda d: d['iem']['abbreviation'], axis=1)
    biomarker_table = biomarker_table[
        ['met_abbreviation', 'disease_abbreviation', 'value', 'normalConcentration', 'rangeConcentration']]

    biomarker_table['normalConcentrationUp'] = biomarker_table.apply(
        lambda d: float(d['normalConcentration'].split('-')[1]) if '-' in d['normalConcentration'] else None, axis=1)
    biomarker_table['normalConcentrationDown'] = biomarker_table.apply(
        lambda d: float(d['normalConcentration'].split('-')[0]) if '-' in d['normalConcentration'] else None, axis=1)
    biomarker_table['rangeConcentrationUp'] = biomarker_table.apply(
        lambda d: float(d['rangeConcentration'].split('-')[1]) if '-' in d['rangeConcentration'] else None, axis=1)
    biomarker_table['rangeConcentrationDown'] = biomarker_table.apply(
        lambda d: float(d['rangeConcentration'].split('-')[0]) if '-' in d['rangeConcentration'] else None, axis=1)

    # Add Metabolite ID
    biomarker_table = biomarker_table.join(metabolite_table.set_index('abbreviation'), on='met_abbreviation')
    biomarker_table = biomarker_table[
        ['met_abbreviation', 'disease_abbreviation', 'value', 'normalConcentrationUp', 'normalConcentrationDown',
         'rangeConcentrationUp', 'rangeConcentrationDown', 'id']]
    biomarker_table = biomarker_table.rename(columns={'id': 'met_id'})

    # Add Disease ID
    biomarker_table = biomarker_table.join(disease_table.set_index('abbreviation'), on='disease_abbreviation')
    biomarker_table = biomarker_table[
        ['value', 'normalConcentrationUp', 'normalConcentrationDown',
         'rangeConcentrationUp', 'rangeConcentrationDown', 'met_id', 'id']]
    biomarker_table = biomarker_table.rename(columns={'id': 'disease_id'})

    biomarker_table['relation'] = biomarker_table.apply(
        lambda d: dataclasses.asdict(
            BiomarkerRelation(
                id=_add_relation(t=RELATION_TYPES[10], head_id=int(d['met_id']), tail_id=int(d['disease_id'])),
                head_id=int(d['met_id']),
                tail_id=int(d['disease_id']),
                value=d['value'],
                normalConcentrationUp=d['normalConcentrationUp'],
                normalConcentrationDown=d['normalConcentrationDown'],
                rangeConcentrationUp=d['rangeConcentrationUp'],
                rangeConcentrationDown=d['rangeConcentrationDown']
            )
        ),
        axis=1
    )
    return biomarker_table['relation'].tolist()


def load_vmh_biomarker_relations():
    _update_a_relation_list(_extract_vmh_biomarker_relations(), BIOMARKER_FILE_NAME)


def _get_met_comp_pairs():
    comp_table = pd.DataFrame(get_raw_compartments())
    raw_metabolite_table = pd.DataFrame(get_raw_metabolites())
    compartment_table = pd.DataFrame(get_compartments())
    metabolite_table = pd.DataFrame(get_metabolites())

    # Clean Comp to Met Table
    new_comp_table = []
    for i in range(comp_table.shape[0]):
        for met_id in comp_table.iloc[i]['mets']:
            new_comp_table.append({'comp_name': comp_table.iloc[i]['name'], 'raw_met_id': met_id})
    comp_table = pd.DataFrame(new_comp_table)

    comp_table = comp_table.join(raw_metabolite_table.set_index('met_id'), on='raw_met_id')
    comp_table = comp_table[['comp_name', 'abbreviation']]

    # Add Compartment ID
    comp_table = comp_table.join(compartment_table.set_index('name'), on='comp_name')
    comp_table = comp_table.rename(columns={'id': 'comp_id'})
    comp_table = comp_table[['comp_id', 'abbreviation']]

    # Add Metabolite ID
    comp_table = comp_table.join(metabolite_table.set_index('abbreviation'), on='abbreviation')
    comp_table = comp_table.rename(columns={'id': 'met_id'})
    comp_table = comp_table[['comp_id', 'met_id']]

    return comp_table


def _extract_vmh_comp_has_met_relations():
    pairs = _get_met_comp_pairs()
    relations = pairs.apply(
        lambda d: dataclasses.asdict(
            CompHasMetRelation(
                id=_add_relation(t=RELATION_TYPES[11], head_id=int(d['met_id']), tail_id=int(d['comp_id'])),
                head_id=int(d['met_id']),
                tail_id=int(d['comp_id'])
            )
        ),
        axis=1
    ).tolist()
    return relations


def load_vmh_comp_has_met_relations():
    _update_a_relation_list(_extract_vmh_comp_has_met_relations(), COMP_HAS_MET_FILE_NAME)


def _extract_vmh_met_in_comp_relations():
    pairs = _get_met_comp_pairs()
    relations = pairs.apply(
        lambda d: dataclasses.asdict(
            MetInCompRelation(
                id=_add_relation(t=RELATION_TYPES[12], head_id=int(d['comp_id']), tail_id=int(d['met_id'])),
                head_id=int(d['comp_id']),
                tail_id=int(d['met_id'])
            )
        ),
        axis=1
    ).tolist()
    return relations


def load_vmh_met_in_comp_relations():
    _update_a_relation_list(_extract_vmh_met_in_comp_relations(), MET_IN_COMP_FILE_NAME)


def _extract_hmdad_microbe_cause_disease_relations():
    pairs = _get_met_comp_pairs()
    relations = pairs.apply(
        lambda d: dataclasses.asdict(
            MetInCompRelation(
                id=_add_relation(t=RELATION_TYPES[12], head_id=int(d['comp_id']), tail_id=int(d['met_id'])),
                head_id=int(d['comp_id']),
                tail_id=int(d['met_id'])
            )
        ),
        axis=1
    ).tolist()
    return relations


def load_hmdad_microbe_cause_disease_relations():
    _update_a_relation_list(_extract_hmdad_microbe_cause_disease_relations(), MICROBE_CAUSE_DISEASE_FILE_NAME)
