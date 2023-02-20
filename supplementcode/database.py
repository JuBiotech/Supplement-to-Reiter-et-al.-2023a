####################### Packages #######################
from .core import *

# Core functions
import time
# Pubchempy
import pubchempy as pcp
# Requests
import requests
# Biopython
from Bio.KEGG import REST
from Bio.KEGG.KGML import KGML_parser
# Chembl
from chembl_webresource_client.utils import utils
# Pillow
from PIL import Image
# Selenium
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.wait import WebDriverWait

####################### Functions #######################

def database_get_organisms(path_results):
    """
    Get all organisms.

    Get all organisms of KEGG 

    Parameters
    ----------
    path_results : str
        Raw string to result forder.
    """
    # Create result folder
    path_results = Path(path_results)
    # Read organism database
    organisms = REST.kegg_list('organism').read()
    # Preallocate list
    list_organisms = []
    # Cycle all organisms
    for organism in organisms.rstrip().split('\n'):
        holder = re.split(';|\t', organism)
        if len(holder) != 8:
            list_organisms.append(holder)
    # Create dataframe
    df_organisms = pandas.DataFrame(
        list_organisms, 
        columns = [
            'Identifier', 'KEGG_Code', 
            'Species', 'Domain', 'Kingdom', 
            'Phylum', 'Subphylum'])
    # Save dataframe
    path_organisms = save_df(
        df_organisms, path_results, 
        'KEGG_list_organisms', index = False)
    return

def database_get_metabolites(path_results):
    """
    Get all metabolites.

    Get all metabolites of KEGG 

    Parameters
    ----------
    path_results : str
        Raw string to result forder.
    """
    # Create result folder
    path_results = Path(path_results)
    # Get list of metabolites
    metabolites = REST.kegg_list('compound').read()
    # Preallocate lists
    list_compound = []
    list_error = []
    dict_metabolites = {}
    # Cycle all metabolites
    for metabolite in metabolites.rstrip().split('\n'):
        holder = re.split(';|\t|\n', metabolite)
        compound_id = holder[0].lstrip('cpd:')
        compound_name = holder[1]
        print(compound_id)

        if int(compound_id.strip('C')) < 20:
            formula = None
            charge = None
            weight = None
            CAS = None
            iupac = None
            sid = None
            cid = None
            chembl = None
            inchi = None
            inchi_key = None
            canonical_smiles = None
            isomeric_smiles = None
            _substance = None
            _compound = None
            try:
                # Read compound
                file_compound = REST.kegg_get(compound_id).read()
                current_section_compound = None
                for line_compound in file_compound.rstrip().split('\n'):
                    section_compound = line_compound[:12].strip()
                    # Cycle compound information
                    if not section_compound == '':
                        current_section_compound = section_compound
                    if current_section_compound == 'FORMULA':
                        hold_formula = line_compound[12:].split('  ')
                        formula = hold_formula[0]
                    if current_section_compound == 'MOL_WEIGHT':
                        hold_weight = line_compound[12:].split('  ')
                        weight = hold_weight[0]
                    if current_section_compound == 'DBLINKS':
                        Identifier = line_compound[12:].split('  ')
                        if 'CAS' in Identifier[0]:
                            CAS = Identifier[0].split(': ')[1]
                        if 'PubChem' in Identifier[0]:
                            sid = Identifier[0].split(': ')[1]
                            try:
                                _substance = pcp.Substance.from_sid(sid)
                                cid = _substance.standardized_cid
                                _compound = pcp.Compound.from_cid(cid)
                            except:
                                _substance = None
                                cid = None
                                _compound = None
                            try:
                                iupac = _compound.iupac_name
                            except:
                                iupac = None
                            try:
                                charge = _compound.charge
                            except:
                                charge = None
                            try:
                                inchi = _compound.inchi
                            except:
                                inchi = None
                            try:
                                inchi_key = _compound.inchikey
                            except:
                                inchi_key = None
                            try:
                                canonical_smiles = _compound.canonical_smiles
                            except:
                                canonical_smiles = None
                            try:
                                isomeric_smiles = _compound.isomeric_smiles
                            except:
                                isomeric_smiles = None
                            
                        if 'ChEMBL' in Identifier[0]:
                            chembl = Identifier[0].split(': ')[1]
                # Add compound information to final list
                list_compound.append(
                    [compound_id, compound_name, 
                    formula, charge, weight, 
                    CAS, iupac, 
                    sid, cid, chembl, 
                    inchi, inchi_key, canonical_smiles, isomeric_smiles]
                    )
                
            except Exception as error_compound:
                print(Exception)
                list_error.append([compound_id, compound_name, error_compound])
    # Create dataframe
    dict_metabolites['Valid'] = pandas.DataFrame(
        list_compound, 
        columns = [
            'compound_id','compound_name',
            'Molecular formula','Charge','Molecular weight',
            'CAS','IUPAC',
            'PubChemSID','PubChemCID','ChEMBL',
            'InChi', 'InChi_key', 'Canonical SMILES', 'Isomeric SMILES']
            )
    dict_metabolites['Error'] = pandas.DataFrame(list_error, columns = ['compound_id','compound_name','Error'])
    # Save dataframe
    path_metabolites = save_dict(dict_metabolites, path_results, 'KEGG_list_compound', single_files = False, index = False)
    return

def database_get_metabolite_information(path_data):
    """
    Get metabolite information.

    Get list of pKa, logD and logP. 

    Parameters
    ----------
    path_data : str    
        Raw string to KEGG_list_compound.xlsx file.  
    """
    # Read dataframe
    path_data = Path(path_data)
    # Parse dataframe
    df_raw = pandas.read_excel(path_data)
    # Select molecules with sufficient information
    df_new = df_raw.dropna(subset = ['ChEMBL']).copy()
    from chembl_webresource_client.new_client import new_client
    molecule = None
    molecule = new_client.molecule
    # Set ChEMBL parameter
    key = 'molecule_properties'
    # Cycle list of smiles in dataframe
    for row, item in df_new.iterrows():
        # Preallocation
        molecule_current = None
        # Get ChEMBL-ID, first ID if multiple IDs are given
        chembl = str(df_new.at[row,'ChEMBL']).split(' ')[0]
        # Access ChEMBL-Database
        molecule_current = molecule.get(chembl)
        keys_properties = molecule_current[key].keys()
        # Access acidic pKa
        if 'cx_most_apka' in keys_properties:
            pKa_acidic = molecule_current[key]['cx_most_apka']
        elif ('acd_most_apka' in keys_properties) & ('cx_most_apka' not in keys_properties):
            pKa_acidic = molecule_current[key]['acd_most_apka']
        else:
            pKa_acidic = None
        # Access basic pKa
        if 'cx_most_bpka' in keys_properties:
            pKa_basic = molecule_current[key]['cx_most_bpka']
        elif ('acd_most_bpka' in keys_properties) & ('cx_most_bpka' not in keys_properties):
            pKa_basic = molecule_current[key]['acd_most_bpka']
        else:
            pKa_basic = None
        # Access logD
        if 'cx_logd' in keys_properties:
            logD = molecule_current[key]['cx_logd']
        elif ('acd_logd' in keys_properties) & ('cx_logd' not in keys_properties):
            logD = molecule_current[key]['acd_logd']
        else:
            logD = None
        # Access logP
        if 'cx_logp' in keys_properties:
            logP = molecule_current[key]['cx_logp']
        elif ('acd_logp' in keys_properties) & ('cx_logp' not in keys_properties):
            logP = molecule_current[key]['acd_logp']
        else:
            logP = None
        # Access alogP
        if 'alogp' in keys_properties:
            alogP = molecule_current[key]['alogp']
        else:
            alogP = None
        # Access species
        if 'molecular_species' in keys_properties:
            molecular_species = str(molecule_current[key]['molecular_species']).lower()
        else:
            molecular_species = None

        print(df_new.at[row,'compound_id'], pKa_acidic, pKa_basic, logD, logP, alogP, molecular_species)
        # Allocate pKa-Value
        df_raw.at[row,'pKa_acidic'] = pKa_acidic
        df_raw.at[row,'pKa_basic'] = pKa_basic
        df_raw.at[row,'logD'] = logD
        df_raw.at[row,'logP'] = logP
        df_raw.at[row,'alogP'] = alogP
        df_raw.at[row,'molecular_species'] = molecular_species
    # Save dataframe
    save_df(df_raw, path_data.parent, 'KEGG_list_pKa', index = False)
    return

def database_get_metabolite_class(path_results):
    # Create result folder
    path_results = Path(path_results)
    list_brites = [
        ('br:br08001','Compounds with biological roles'),
        ('br:br08002','Lipids'),
        ('br:br08003','Phytochemical compounds'),
        ('br:br08021','Glycosides'),
    ]
    
    reg = '(^\s+)'
    reg_class = '( \[Fig])'
    reg_compound = re.compile('(C[0-9]+)')
    reg_orthologies = '( \[.+?\])'
    list_information = []
    for brite in list_brites:
        brite_single = REST.kegg_get(brite[0])
        for line in brite_single:
            level = line[0].upper()
            name_raw = line[1:].split('\n')[0]
            name_split = re.split(reg, name_raw)[-1]
            if brite[0] in [
                'br:br08001',
                'br:br08002',
                'br:br08003',
                'br:br08021',
            ]:
                compound_function = None
                
                if level == 'A':
                    metabolite_group = line[1:].split('\n')[0]
                if level == 'B':
                    metabolite_class = line[1:].split('\n')[0]
                    metabolite_class = re.split(reg, metabolite_class)[-1]
                    metabolite_class = re.split(reg_class, metabolite_class)[0]
                if level == 'C':
                    metabolite_subclass = line[1:].split('\n')[0]
                    metabolite_subclass = re.split(reg, metabolite_subclass)[-1]
                    metabolite_subclass = re.split(reg_class, metabolite_subclass)[0]
                    metabolite_subclass = re.split(reg_orthologies, metabolite_subclass)[0]
                if level == 'D':
                    metabolite_compound = line[1:].split('\n')[0]
                    metabolite_compound = re.split(reg, metabolite_compound)[-1]
                    compound_id = metabolite_compound[:6]
                    compound_name = metabolite_compound[6:]
                    compound_name = re.split(reg,compound_name)[-1]
                    if reg_compound.search(compound_id):
                        list_information.append(
                        [
                            brite[0], brite[1],
                            metabolite_group,
                            metabolite_class,
                            metabolite_subclass,
                            compound_id,
                            compound_name,
                        ]
                        )

    df_classes = pandas.DataFrame(list_information, columns = ['brite_id','brite_name','group','class','subclass','compound_id','compound_name'])
    save_df(df_classes, path_results, 'KEGG_list_class', index = False)
    return

def database_get_pathway_information(path_results, list_organisms):
    """
    Get pathway information.

    Get organism specific pathways, compounds, enzymes, genes and reactions. 

    Parameters
    ----------
    path_results : str    
        Raw string to result folder.
    list_organisms : list
        List of KEGG organism identifier
    """
    path_results = Path(path_results)
    # Cycle organisms
    for id_organism in list_organisms:
        # Preallocation
        dict_organism = {}
        list_compound = []
        list_gene = []
        list_reactions = []
        list_network_nodes = []
        list_network_genes = []
        list_network_reactions = []
        list_network_maplink = []
        # Read pathways
        if id_organism.lower() == 'reference':
            id_organism = 'ko'
        file_organism = REST.kegg_list('pathway', id_organism).read()
        # Cycle pathways
        for line_organism in file_organism.rstrip().split('\n'):
            if id_organism == 'ko':
                # Set section label
                section_label = 'ORTHOLOGY'
                # Split information
                holder = re.split('\t', line_organism)
                # Allocate pathway information
                id_pathway, name_pathway = holder
                name_organism = 'reference'
            else:
                # Set section label
                section_label = 'GENE'
                # Split information
                holder = re.split(';|\t| - ', line_organism)
                # Delete additional name
                if len(holder) == 4:
                    holder_new = [holder[0], holder[1]+' - '+holder[2], holder[3]]
                    holder = holder_new
                # Allocate pathway information
                id_pathway, name_pathway, name_organism = holder
            # Print information
            print(f'Organism: {id_organism}; Pathway: {id_pathway}', end="\r")
            # Read pathway information
            file_pathway = REST.kegg_get(id_pathway).read()
            id_pathway = re.sub('(^[^:]*:)','', id_pathway)     
            # Set initial section
            current_section_pathway = None
            # Cycle compounds of pathway
            for line_pathway in file_pathway.rstrip().split('\n'):
                # Get section information
                section_pathway = line_pathway[:12].strip()
                # Set section
                if not section_pathway == '':
                    current_section_pathway = section_pathway
                ### Compound ###
                if current_section_pathway == 'COMPOUND':
                    # Reset compound information
                    compound_id = None
                    compound_name = None
                    try:
                        # Split information
                        compound_id, compound_name = re.split('\s{2,}', line_pathway[12:])
                    except:
                        compound_id = line_pathway[12:].strip()
                        compound_name = None
                    # Append compound
                    list_compound.append(
                        [id_organism, name_organism, 
                        id_pathway, name_pathway, 
                        compound_id, compound_name]
                        )
                ### Orthology ###
                if current_section_pathway == section_label:
                    # Reset compound information
                    gene_id = None
                    gene_name = None
                    # Split information
                    gene_id, gene_name = re.split('\s{2,}', line_pathway[12:])
                    # Extract gene and enzyme information
                    holder_gene = re.split('(;|\[[ECKO:].*?\])', gene_name)
                    holder_gene = [x.strip(' |;') for x in holder_gene]
                    holder_gene = [x for x in holder_gene if x]
                    # Get enzyme code
                    list_enzyme_code = [item for item in holder_gene if '[EC' in item]
                    if len(list_enzyme_code) == 1:
                        enzyme_code = list_enzyme_code[0].strip('[]').lstrip('EC:').split()
                    else:
                        enzyme_code = None
                    # Get enzyme name
                    list_kegg_orthology = [item for item in holder_gene if '[KO' in item]
                    list_names = [item for item in holder_gene if item not in (list_enzyme_code+list_kegg_orthology)]
                    if len(list_names) == 2:
                        gene_abb = list_names[0]
                        gene_name = list_names[1]
                    else:
                        gene_abb = None
                        gene_name = list_names[0]
                    ### Enzyme ###
                    if enzyme_code:
                        for enzyme in enzyme_code:
                            # Append enzyme
                            list_gene.append([id_organism, name_organism, id_pathway, name_pathway, gene_id, gene_name, gene_abb, enzyme])   
                            try:
                                # Read enzyme information
                                file_enzyme = REST.kegg_get(enzyme).read()
                                # Set initial section
                                current_section_enzyme = None
                                # Cycle compounds of pathway
                                for line_enzyme in file_enzyme.rstrip().split('\n'):
                                    # Get section information
                                    section_enzyme = line_enzyme[:12].strip()
                                    # Set section
                                    if not section_enzyme == '':
                                        current_section_enzyme = section_enzyme

                                    # Get substrates and products
                                    compound_type = None
                                    holder_substrate = None
                                    holder_product = None
                                    ### Substrate ###
                                    if current_section_enzyme == 'SUBSTRATE':
                                        compound_type = 'substrate'
                                        holder_substrate = re.split('(\[[CPD:].*?\])', line_enzyme[12:])
                                        holder_substrate = [item for item in holder_substrate if '[CPD' in item]
                                        if holder_substrate:
                                            holder_substrate = holder_substrate[0].strip('[]')
                                            substrates = [re.sub('(^[^:]*:)','', item) for item in holder_substrate.split(' ')]
                                            for substrate in substrates:
                                                list_reactions.append([id_organism, name_organism, id_pathway, name_pathway, gene_id, gene_name, gene_abb, enzyme, compound_type, substrate])
                                    ### Substrate ###
                                    if current_section_enzyme == 'PRODUCT':
                                        compound_type = 'product'
                                        holder_product = re.split('(\[[CPD:].*?\])', line_enzyme[12:])
                                        holder_product = [item for item in holder_product if '[CPD' in item]
                                        if holder_product:
                                            holder_product = holder_product[0].strip('[]')
                                            products = [re.sub('(^[^:]*:)','', item) for item in holder_product.split(' ')]
                                            for product in products:
                                                list_reactions.append([id_organism, name_organism, id_pathway, name_pathway, gene_id, gene_name, gene_abb, enzyme, compound_type, product])
                            except:
                                None
            try:
                # Read pathway kgml
                kgml_pathway = REST.kegg_get(id_pathway, 'kgml')
                kgml_parsed = KGML_parser.read(kgml_pathway)
                # Cycle nodes and genes
                for entry_id in kgml_parsed.entries:
                    entry = kgml_parsed.entries[entry_id]
                    entry_type = entry.type
                    entry_names = [re.sub('(^[^:]*:)','', item) for item in entry.name.split(' ')]
                    if entry_type in ['compound', 'glycan', 'map']:
                        entry_pos = entry.graphics[0].centre
                        for name in entry_names:
                            list_network_nodes.append(
                                [id_organism, name_organism, id_pathway, name_pathway, 
                                entry_id, entry_type, name, entry_pos[0], entry_pos[1]]
                            )

                    elif entry_type in ['gene', 'ortholog']:
                        entry_reaction = [re.sub('(^[^:]*:)','', item) for item in entry.reaction.split(' ')]
                        entry_pairs = [item for item in itertools.product(*[entry_names, entry_reaction])]
                        for entry_pair in entry_pairs:
                            list_network_genes.append(
                                [id_organism, name_organism, id_pathway, name_pathway, 
                                entry_id, entry_type, entry_pair[0], entry_pair[1]]
                                )
                    else:
                        None

                # Cycle reactions
                for reaction in kgml_parsed.reactions:
                    reaction_id = reaction.id
                    reaction_type = reaction.type
                    reaction_names = [re.sub('(^[^:]*:)','', item) for item in reaction.name.split(' ')]
                    # Substrates
                    reaction_substrates = []
                    for subs in reaction.substrates:
                        names = str(subs.name).split(' ')
                        for name in names:
                            reaction_substrates.extend([re.sub('(^[^:]*:)','', name)])
                    # Products
                    reaction_products = []
                    for prod in reaction.products:
                        names = str(prod.name).split(' ')
                        for name in names:
                            reaction_products.extend([re.sub('(^[^:]*:)','', name)])
                    reaction_pairs = [item for item in itertools.product(*[reaction_names, reaction_substrates, reaction_products])]
                    for reaction_pair in reaction_pairs:
                        list_network_reactions.append(
                            [id_organism, name_organism, id_pathway, name_pathway, 
                            reaction_id, reaction_pair[0], reaction_pair[1], reaction_pair[2], reaction_type])
                
                # Cycle relations
                for relation in kgml_parsed.relations:
                    relation_type = relation.type
                    entry_1 = relation.entry1 
                    entry_2 = relation.entry2
                    entry_1_id = entry_1.id
                    entry_2_id = entry_2.id
                    entry_1_type = entry_1.type
                    entry_2_type = entry_2.type
                    if relation.subtypes:
                        relation_subtypes = relation.subtypes
                        for subtype in relation_subtypes:
                            subtype_name = subtype[0]
                            subtype_id = subtype[1]
                            list_network_maplink.append(
                                [id_organism, name_organism, id_pathway, name_pathway, 
                                relation_type, entry_1_type, entry_2_type, entry_1_id, entry_2_id, subtype_name, subtype_id])
            except:
                None
                    
        # Create dictionary
        dict_organism['compounds'] = pandas.DataFrame(
            list_compound, columns = ['organism_id','organism_name', 'pathway_id', 'pathway_name', 'compound_id', 'compound_name'])
        dict_organism['genes']  = pandas.DataFrame(
            list_gene, columns = ['organism_id','organism_name', 'pathway_id', 'pathway_name', 'gene_id', 'gene_name', 'gene_abb', 'enzyme_code'])
        dict_organism['reactions']  = pandas.DataFrame(
            list_reactions, columns = ['organism_id','organism_name', 'pathway_id', 'pathway_name', 'gene_id', 'gene_name', 'gene_abb', 'enzyme_code', 'compound_type', 'compound_id'])
        dict_organism['network_nodes'] = pandas.DataFrame(
            list_network_nodes, columns = ['organism_id','organism_name', 'pathway_id', 'pathway_name', 'id_pos', 'type', 'name_node', 'pos_x', 'pos_y']).drop_duplicates().reset_index(drop=True)
        dict_organism['network_genes'] = pandas.DataFrame(
            list_network_genes, columns = ['organism_id','organism_name', 'pathway_id', 'pathway_name', 'id_pos', 'type', 'gene', 'reaction']).drop_duplicates().reset_index(drop=True)
        dict_organism['network_reactions']  = pandas.DataFrame(
            list_network_reactions, columns = ['organism_id','organism_name', 'pathway_id', 'pathway_name', 'id_pos', 'reaction_id', 'substrate', 'product', 'type']).drop_duplicates().reset_index(drop=True)
        dict_organism['network_maplink'] = pandas.DataFrame(
            list_network_maplink, columns = ['organism_id','organism_name', 'pathway_id', 'pathway_name', 'relation_type', 'entry_1_type', 'entry_2_type', 'entry_1_id', 'entry_2_id', 'subtype_name', 'subtype_id']).drop_duplicates().reset_index(drop=True)
        # Save dictionary
        save_dict(dict_organism, path_results, f'{id_organism}', single_files = False)
    return

def database_predict_fragmentation_offline(
    path_metabolite_list, path_results, path_organism,
    path_exe_predict, 
    path_file_parameter_pos, path_file_parameter_neg,
    path_file_config_pos, path_file_config_neg,
    structure_key = 'inchi', modes = ['Pos','Neg'],
    ):
    """
    Predict fragmentation pattern offline.

    Predict fragmentation pattern of metabolites with CFM-ID base code.

    Parameters
    ----------
    path_metabolite_list : str
        Raw string to KEGG_list_pKa.xlsx file.
    path_results : str
        Raw string to results folder.
    path_exe_predict : str
        Raw string to cfm-predict.exe.
    path_file_parameter_pos : str
        Raw string to param_output0.log in params_metab_se_cfm.
    path_file_parameter_neg : str
        Raw string to param_output0.log in negative_metab_se_cfm\negative_se_params.
    path_file_config_pos : str
        Raw string to param_config.txt in metab_se_cfm.
    path_file_config_neg : str
        Raw string to param_config.txt in negative_metab_se_cfm.
    structure_key : str
        Structure key (Inchi or SMILES).
    modes : list, default ['Pos','Neg']
        List of ionization modes.
    """
    # Set standard prediction parameter
    prob_thresh = '0.001'
    annotate_fragments = '1'
    apply_postproc = '1'
    suppress_exceptions = '0'
    # Set regular expression for command line output
    regex = r'(.+?) (.+?) (.+) (\(.+)'
    # Parse dataframe
    df_molecules = pandas.read_excel(Path(path_metabolite_list))
    # Predict only metabolites with key
    df_molecules = df_molecules.dropna(subset = ['InChi','Canonical SMILES']).copy()
    # Select molecules with mol weight < 1500
    df_molecules = df_molecules[df_molecules['Molecular weight'] < 1500]
    # If organism is provided
    try:
        dict_organism = pandas.read_excel(Path(path_organism), sheet_name = None)
    except:
        dict_organism = {}

    if dict_organism:
        df_reactions_enzyme = dict_organism['reactions'].copy()
        df_reactions_kgml = dict_organism['network_reactions'].copy()
        set_organism = set(df_reactions_enzyme['compound_id'])|set(df_reactions_kgml['substrate'])|set(df_reactions_kgml['product'])
        df_molecules = df_molecules[df_molecules['compound_id'].isin(set_organism)].copy()

    # Select key
    if structure_key.lower() == 'inchi':
        structure_key = 'InChi'
    else:
        structure_key = 'Canonical SMILES'
    # Cycle modes
    for mode in modes:
        # Reset dataframe
        df_molecule_hold = df_molecules.copy()
        # Set ionization mode parameter
        if mode == 'Pos':
            model = path_file_parameter_pos
            config = path_file_config_pos
        else:
            model = path_file_parameter_neg
            config = path_file_config_neg

        # Get already predicted metabolites
        list_predicted = [item for item in os.listdir(path_results) if (item.endswith('.txt') & (mode in item))]
        list_missing = [abb for abb in df_molecule_hold['compound_id'] if abb+'_'+mode+'.txt' not in list_predicted]
        df_molecule_hold = df_molecule_hold[df_molecule_hold['compound_id'].isin(list_missing)].copy().reset_index()
        # Cycle all compounds
        for idx, row in df_molecule_hold.iterrows():
            # Get metabolite information
            abb = df_molecule_hold.at[idx,'compound_id']
            # Print molecules to predict
            try:
                print(f'Current: {abb}. To predict in {mode}: {len(df_molecule_hold)-len(df_molecule_hold[:idx])}')
            except:
                None
            # Get key information
            compound_key = df_molecule_hold.at[idx, structure_key]
            # Set compound command
            path_command_predictions = [path_exe_predict,compound_key,prob_thresh,model,config,annotate_fragments]
            try:
                # Predictions
                output_predictions = subprocess.check_output(path_command_predictions).decode('utf-8').rstrip().split('\r\n')
                save_list(output_predictions, Path(path_results), f'{abb}_{mode}')
            except:
                print('No prediction.')
            time.sleep(2)
    return

def database_predict_fragmentation_online(
    path_metabolite_list, path_results, path_organism, path_driver, 
    url, structure_key = 'inchi', modes = ['Pos','Neg'],
    ):
    """
    Predict fragmentation pattern online.

    Predict fragmentation pattern of metabolites with CFM-ID web service.

    Parameters
    ----------
    path_metabolite_list : str
        Raw string to KEGG_list_pKa.xlsx file.
    path_results : str
        Raw string to results folder.
    path_driver : str
        Raw string to chromedriver.exe.
    url : str
        Raw string to prediction website (e.g. http://cfmid3.wishartlab.com/predict)
    structure_key : str
        Structure key (Inchi or SMILES).
    modes : list, default ['Pos','Neg']
        List of ionization modes.
    """
    # Parse dataframe
    df_molecules = pandas.read_excel(path_metabolite_list)
    # Select molecules with sufficient information
    df_molecules = df_molecules.dropna(subset = ['InChi','Canonical SMILES']).copy()
    # Select molecules with mol weight < 1500
    df_molecules = df_molecules[df_molecules['Molecular weight'] < 1500]
    # If org is provided
    try:
        dict_organism = pandas.read_excel(Path(path_organism), sheet_name = None)
    except:
        dict_organism = {}

    if dict_organism:
        df_reactions_enzyme = dict_organism['reactions'].copy()
        df_reactions_kgml = dict_organism['network_reactions'].copy()
        set_organism = set(df_reactions_enzyme['compound_id'])|set(df_reactions_kgml['substrate'])|set(df_reactions_kgml['product'])
        df_molecules = df_molecules[df_molecules['compound_id'].isin(set_organism)].copy()
    
    if structure_key.lower() == 'inchi':
        structure_key = 'InChi'
    else:
        structure_key = 'Canonical SMILES'
        
    # Set driver engine and settings
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(executable_path = path_driver, options = options)
    # Cylce modes
    for mode in modes:
        # Reset dataframe
        df_molecule_hold = df_molecules.copy()
        if mode == 'Pos':
            adducts = ['[M+H]+']
            keys_siteindex = [0]
        else:
            adducts = ['[M-H]-']
            keys_siteindex = [0]

        # Get already predicted metabolites
        list_predicted = [item for item in os.listdir(path_results) if (item.endswith('.txt') & (mode in item))]
        list_missing = [abb for abb in df_molecule_hold['compound_id'] if abb+'_'+mode+'.txt' not in list_predicted]
        df_molecule_hold = df_molecule_hold[df_molecule_hold['compound_id'].isin(list_missing)].copy().reset_index()
        # Cycle all compounds
        for idx, row in df_molecule_hold.iterrows():
            # Get metabolite information
            abb = df_molecule_hold.at[idx,'compound_id']
            metabolite = df_molecule_hold.at[idx,'compound_name']
            # Print molecules to predict
            try:
                print(f'Current: {abb}. To predict in {mode}: {len(df_molecule_hold)-len(df_molecule_hold[:idx])}')
            except:
                None
            # Cycle adducts
            for idx1, idx2 in zip(adducts, keys_siteindex):
                # Open page and wait
                driver.get(url)
                time.sleep(3)
                # Find fields and fill information
                key_field = driver.find_element_by_id('predict_query_compound')
                key_field.send_keys(df_molecule_hold.loc[idx, structure_key])
                ion_field = driver.find_element_by_id('predict_query_ion_mode')
                ion_field.send_keys(mode)
                adduct_field = Select(driver.find_element_by_id('predict_query_adduct_type'))
                adduct_field.select_by_index(idx2)
                # Submit
                submit_button = driver.find_element_by_id('query-submit')
                submit_button.click()
                try:
                    # Wait for progress page to appear
                    time.sleep(3)
                    # Waiting for 600s is not sufficient for very large molecules
                    progressbar=driver.find_element_by_class_name('processing-box')
                    WebDriverWait(driver, 20).until(EC.invisibility_of_element_located(progressbar))
                    WebDriverWait(driver, 1000).until(EC.title_contains('Results'))
                    time.sleep(3)
                    # Select prediction tab
                    try:
                        tab_computed = driver.find_element_by_id('computed-tab')
                        tab_computed.click()
                    except:
                        None
                    # Download results
                    download_button = driver.find_element_by_class_name('btn.btn-xs.btn-download')
                    download_button.click()
                    # Wait for result page to appear
                    time.sleep(3)
                    # Download results
                    url_download = driver.current_url
                    resultfile_text = requests.get(url_download)
                    # Save results
                    resultfile_name = f'{abb}_{mode}.txt'
                    p_resultfile = Path(path_results).joinpath(resultfile_name) 
                    text_document = open(f'{p_resultfile}','w+')
                    text_document.write(resultfile_text.text)
                    text_document.close()
                except:
                    None
    driver.quit()
    return
