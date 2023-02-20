####################### Packages #######################

from .core import *

####################### Functions #######################

def development(
    path_project, name_method,
    path_database_literatur, path_database_inorganics, path_database_predictions,
    path_database_metabolites, path_database_pathways, 
    organisms = [], pathways = [], drop_sumpeaks = False, drop_sumpeaks_no = 3
    ):
    """
    Automated method development.

    Main function of dsfiapy method development to create dilute-and-shoot flow-injection-analysis tandem mass spectrometry methods.

    Parameters
    ----------
    path_project : str
        Raw string to results folder.
    path_database_literatur : str
        Raw string to literature data.
    path_database_inorganics : str
        Raw string to inorganics data.
    path_database_predictions : str
        Raw string to predictions folder.
    path_database_metabolites : str
        Raw string to metabolite information file.
    path_database_pathways : str
        Raw string to pathway folder.
    organisms : list
        List of KEGG organism identifier.
    drop_sumpeaks : bool, default False
        Drop convoluted metabolite mass transitions.
    drop_sumpeaks_no : int, default 3
        If drop_sumpeaks == True, drop convoluted metabolite mass transitions greater equal than int

    Returns
    -------
    dict_methods : dict
        Dictionary with DS-FIA-MS/MS methods.
    """
    # Parse input
    inp = development_parse_input(
        path_project, name_method,
        path_database_literatur, path_database_inorganics, path_database_predictions,
        path_database_metabolites, path_database_pathways, 
        organisms, pathways, drop_sumpeaks,drop_sumpeaks_no
        )
    # Create folder
    inp = development_folder(inp)
    # Parse data
    df_literature, df_inorganics, df_predictions = development_parse_data(inp)
    # Database: Merge data
    df_database = development_create_database(
        df_literature, df_inorganics, df_predictions, inp)
    # Database: Get isobarics
    df_isobarics = development_get_isobarics(df_database)
    # Get MSMS parameter: Extract parameter
    df_mrms_all = development_extract_parameter(df_database, df_isobarics, inp)
    # Select mass transitions
    df_mrms_single, df_isobarics_new = development_select_transitions(df_mrms_all, df_isobarics, inp)
    # Get MSMS parameter: Get convoluted signals
    df_mrms_convoluted, df_mrms_dropped = development_group_convoluted_signals(df_mrms_single, df_isobarics_new, inp)
    # Format methods
    dict_methods = development_format_output(df_mrms_convoluted, inp)
    # Create development resume plot
    df_mrms_single_resume = df_mrms_single.copy()
    development_plot_resume(df_mrms_single_resume, inp)
    # Allocate dictionaries
    dict_libraries = {}
    dict_reports = {}
    dict_libraries['ion_library'] = df_database.copy()
    dict_libraries['isobarics_all'] = df_isobarics.copy()
    dict_libraries['isobarics_selected'] = df_isobarics_new.copy()
    dict_reports['single'] = df_mrms_single.copy()
    dict_reports['convoluted'] = df_mrms_convoluted.copy()
    dict_reports['dropped'] = df_mrms_dropped.copy()
    save_dict(
        dict_libraries, inp['path_development'], 
        'libraries', single_files = False, index = False)
    save_dict(
        dict_reports, inp['path_development'], 
        'reports', single_files = False, index = False) 
    save_dict(
        dict_methods, inp['path_method'], 
        'methods', single_files = False, index = False)
    return dict_methods
    
def development_parse_input(
        path_project, name_method,
        path_database_literatur, path_database_inorganics, path_database_predictions,
        path_database_metabolites, path_database_pathways, 
        organisms, pathways, drop_sumpeaks, drop_sumpeaks_no
        ):
    """
    Parse input.

    Parse user input and generate dictionary for easy access.

    Parameters
    ----------
    path_project : str
        Raw string to results folder.
    path_database_literatur : str
        Raw string to literature data.
    path_database_inorganics : str
        Raw string to inorganics data.
    path_database_predictions : str
        Raw string to predictions folder.
    path_database_metabolites : str
        Raw string to metabolite information file.
    path_database_pathways : str
        Raw string to pathway folder.
    organisms : list
        List of KEGG organism identifier.
    drop_sumpeaks : bool, default False
        Drop convoluted metabolite mass transitions.
    drop_sumpeaks_no : int, default 3
        If drop_sumpeaks == True, drop convoluted mass transitions greater equal than int

    Returns
    -------
    inp : dict
        Dictionary with user input.
    """
    inp = {}
    # Set paths
    inp['path_project'] = Path(path_project)
    inp['name_method'] = name_method
    inp['path_literature'] = Path(path_database_literatur)
    inp['path_inorganics'] = Path(path_database_inorganics)
    inp['path_predictions'] = Path(path_database_predictions)
    inp['path_metabolites'] = Path(path_database_metabolites)
    inp['path_pathways'] = Path(path_database_pathways)
    # Set parameter
    inp['organisms'] = organisms
    inp['pathways'] = pathways
    inp['drop_sumpeaks'] = drop_sumpeaks
    inp['drop_sumpeaks_no'] = drop_sumpeaks_no
    # Set plotting parameter
    inp['figsize'] = (6,5)
    inp['labelsize'] = 14
    return inp

def development_folder(inp):
    """
    Create folder.

    Create folder for method development result.

    Parameters
    ----------
    inp : dict
        Input parameter.

    Returns
    -------
    inp : dict
        Input parameter.
    """
    # Create folder
    inp['path_method'] = create_folder(inp['path_project'], inp['name_method'])
    inp['path_development'] = create_folder(inp['path_method'], 'development')
    return inp

def development_parse_data(inp):
    """
    Parse data of input.

    Parse raw data based on user input.

    Parameters
    ----------
    inp : dict
        Input dictionary.

    Returns
    -------
    df_literature : dataframe
        Dataframe with literature data.
    df_inorganics : dataframe
        Dataframe with inorganics data.
    df_predictions : dataframe
        Dataframe with prediction data.
    """
    # Get metabolite list
    df_metabolites = development_parse_data_metabolite_list(inp)
    # Set mode for metabolites
    df_metabolites = development_parse_data_set_metabolite_mode(inp, df_metabolites)
    # Parse literature
    df_literature, df_metabolites = development_parse_data_literature(inp, df_metabolites)
    # Parse inorganics
    df_inorganics, df_metabolites = development_parse_data_inorganics(inp, df_metabolites)
    # Parse predictions
    df_predictions = development_parse_data_predictions(inp, df_metabolites)
    return df_literature, df_inorganics, df_predictions

def development_parse_data_metabolite_list(inp):
    """
    Parse and prepare metabolite list.

    Parsing and preprocessing of organism metabolite list.

    Parameters
    ----------
    inp : dict
        Input dictionary.

    Returns
    -------
    df_metabolites_new : dataframe
        Dataframe with preprocessed metabolite information.
    """
    # Parse metabolite information
    df_metabolites = pandas.read_excel(inp['path_metabolites'])
    # Filter metabolite size
    df_metabolites = df_metabolites[
        (df_metabolites['Molecular weight'] <= 1500) &
        (df_metabolites['Molecular weight'] >= 30) &
        (~df_metabolites['Molecular weight'].isna())
        ].copy()
    # Set value type
    df_metabolites['Molecular weight'] = df_metabolites['Molecular weight'].fillna(0).apply(lambda x: round(float(x)))
    df_metabolites['Charge'] = df_metabolites['Charge'].fillna(0).apply(lambda x: round(float(x)))
    # Filter metabolite pKa acidic or basic
    df_metabolites = df_metabolites[(~df_metabolites['pKa_acidic'].isna()) | (~df_metabolites['pKa_basic'].isna())].copy()
    # Select metabolites, all possible or based on model
    
    # Preallocate set
    set_metabolites = set()
    # Preallocate dataframes
    df_compounds = pandas.DataFrame()
    df_genes = pandas.DataFrame()
    df_reactions = pandas.DataFrame()
    # Preallocate dictionary
    dict_mapping = {}
    # Cycle organisms
    for organism in inp['organisms']:
        # Parse organism metabolite set
        dict_metabolites = pandas.read_excel(inp['path_pathways'].joinpath(f'{organism}.xlsx'), sheet_name = None)
        # Parse information
        df_hold_compounds = dict_metabolites['compounds'].copy()
        df_hold_genes = dict_metabolites['genes'].copy()

        # Selection of metabolites
        df_hold_reactions = dict_metabolites['reactions'].copy()
        set_metabolites_hold = set(df_hold_reactions['compound_id'])
            
        # Specific pathway selection
        if inp['pathways']:
            df_hold_reactions = df_hold_reactions[df_hold_reactions['pathway_id'].isin(inp['pathways'])].copy()

        print(f'Metabolites in {organism}: {len(set_metabolites_hold)}')
        # Append sets
        set_metabolites = set_metabolites|set_metabolites_hold
        # Append dataframes
        df_compounds = df_compounds.append(df_hold_compounds)
        df_genes = df_genes.append(df_hold_genes)
        df_reactions = df_reactions.append(df_hold_reactions)

    # Allocate dataframes to dictionary
    dict_mapping['compounds'] = df_compounds.copy()
    dict_mapping['genes'] = df_genes.copy()
    dict_mapping['reactions'] = df_reactions.copy()
    # Save dictionary
    save_dict(dict_mapping, inp['path_development'], 'origin', single_files = False, index = False)
    # Select metabolites
    df_metabolites_new = df_metabolites[df_metabolites['compound_id'].isin(set_metabolites)].copy()
    print(f'Metabolites in database for model: {len(set(df_metabolites_new["compound_id"]))}')
    # Filter enantiomeres after metabolite selection
    df_metabolites_new = development_parse_data_metabolite_list_filter_enantiomeres(df_metabolites_new).copy()
    print(f'Metabolites in database to use after enantiomere selection: {len(set(df_metabolites_new["compound_id"]))}')
    return df_metabolites_new

def development_parse_data_metabolite_list_filter_enantiomeres(df):
    """
    Filter enantiomeres.

    Filter enantiomeres based on name and formula.

    Parameters
    ----------
    df : dataframe
        Dataframe with enantiomeres.

    Returns
    -------
    df_new : dataframe
        Dataframe without enantiomeres.
    """
    df_new = df.copy()
    # Format data
    df_new['Enantiomere Q1'] = df_new['Molecular weight'].apply(lambda x: round(float(x)))
    # Sort values to prefer L enantiomeres
    df_new = df_new.sort_values(by=['compound_name'], ascending = False) # L for D enantiomeres
    # Preallocate lists
    list_keep = []
    list_drop = []
    # Cycle enantiomeres and formular
    for mw, mf in zip(df_new['Enantiomere Q1'], df_new['Molecular formula']):
        df_hold = df_new[(df_new['Enantiomere Q1'] == mw) & (df_new['Molecular formula'] == mf)]
        if len(df_hold) == 1:
            list_keep.extend(list(df_hold['compound_name']))
        if len(df_hold) > 1:
            wordList = list(df_hold['compound_name'])
            for word in wordList:
                if word not in (list_keep+list_drop):
                    list_keep.append(word)
                wordList1 = [w for w in wordList if (sum(a!=b for a,b in zip(word,w)) == 1) & (w not in list_keep) & (w not in list_drop)]
                list_drop.extend(wordList1)
    # Drop enantiomeres
    df_new = df[~df['compound_name'].isin(list_drop)].copy()
    return df_new

def development_parse_data_set_metabolite_mode(inp, df_metabolites):
    """
    Set metabolite mode.

    Set ionization mode based on pKa/pKb value.
    
    Parameters
    ----------
    inp : dict
        Input dictionary.
    df_metabolites : dataframe
        Preprocessed metabolite dataframe

    Returns
    -------
    df_metabolites : dataframe
        Metabolite dataframe with metabolite ionization mode.
    """
    # Calculate pKb of acidic and basic groups
    df_metabolites['pKb_acidic'] = 14-df_metabolites['pKa_acidic'] 
    df_metabolites['pKb_basic'] = 14-df_metabolites['pKa_basic'] 

    # Cycle metabolites
    for index, row in df_metabolites.iterrows():
        # Acids
        if row['molecular_species'] == 'acid': 
            df_metabolites.at[index,'mode'] = 'Neg'
        # Bases
        elif row['molecular_species'] == 'base': 
            df_metabolites.at[index,'mode'] = 'Pos'
        # Zwitterions
        elif row['molecular_species'] == 'zwitterion':
            df_metabolites.at[index,'mode'] = 'Both'
        # Neutral
        else:
            # Only pKa acidic present
            if (~pandas.isnull(row['pKa_acidic'])) & (pandas.isnull(row['pKa_basic'])): 
                df_metabolites.at[index,'mode'] = 'Neg'
            # Only pKa basic present
            elif (pandas.isnull(row['pKa_acidic'])) & (~pandas.isnull(row['pKa_basic'])):
                df_metabolites.at[index,'mode'] = 'Pos'
            # Both pKa present
            else: 
                # pKa of acidic group smaller than pKb of basic group -> Acid
                if (row['pKa_acidic'] < row['pKb_basic']):
                    df_metabolites.at[index,'mode'] = 'Neg'
                # pKa of acidic group bigger than pKb of basic group -> Base
                else:
                    df_metabolites.at[index,'mode'] = 'Pos'

    # Mode for charged molecules with contrary pKs, e.g. Fe3+ is pos but shows low pKs
    df_charged = df_metabolites[df_metabolites['Charge'] != 0].copy()
    # Cycle 
    for index, row in df_charged.iterrows():
        # Check charge
        if row['Charge'] > 0:
            df_metabolites.at[index,'mode'] = 'Pos'
        else:
            df_metabolites.at[index,'mode'] = 'Neg'
    return df_metabolites

def development_parse_data_literature(inp, df_metabolites):
    """
    Parse and prepare literature data.

    Parse and preprocess literature data for database creation.
    
    Parameters
    ----------
    inp : dict
        Input dictionary.
    df_metabolites : dataframe
        Preprocessed metabolite dataframe

    Returns
    -------
    df_literature : dataframe
        Dataframe with parsed literature data.
    df_metabolites : dataframe
        Preprocessed metabolite dataframe with priorization.
    """
    # Parse literature data
    df_literature = pandas.read_excel(inp['path_literature']).drop_duplicates(keep='first')
    df_literature.apply(pandas.to_numeric, errors = 'ignore')
    df_literature = df_literature.round(0)
    df_literature.drop(columns = ['charge_number_Q1','metabolic_group'], inplace = True)
    # Priorization of IBG1 > Literature
    df_ibg = df_literature[df_literature['reference']=='IBG1'].copy()
    df_other = df_literature[
        ~(df_literature['reference']=='IBG1') &
        ~(df_literature['compound_id'].isin(df_ibg['compound_id'].unique()))
        ].copy()
    df_literature = df_ibg.append(df_other, ignore_index = True, sort = False)
    # Select literature metabolites
    df_literature = df_literature[df_literature['compound_id'].isin(df_metabolites['compound_id'].unique())].copy().sort_values(by = 'origin', axis = 0, ascending = True)
    # Mark literature metabolites in dataframe
    df_metabolites['priorization'] = df_metabolites['compound_id'].isin(df_literature['compound_id'])
    for index, row in df_metabolites.iterrows():
        if row['priorization'] == True:
            df_abb = df_literature[df_literature['compound_id'] == row['compound_id']]
            df_metabolites.at[index, 'mode_preferred'] = df_abb['mode'].unique()[0]
            df_metabolites.at[index, 'number_fragments_literature'] = len(df_abb)
    return df_literature, df_metabolites

def development_parse_data_inorganics(inp, df_metabolites):
    """
    Parse and prepare inorganics data.

    Parse and preprocess inorganics data for database creation.
    
    Parameters
    ----------
    inp : dict
        Input dictionary.
    df_metabolites : dataframe
        Preprocessed metabolite dataframe

    Returns
    -------
    df_inorganics : dataframe
        Dataframe with parsed inorganics data.
    df_metabolites : dataframe
        Preprocessed metabolite dataframe with priorization.
    """
    # Parse inorganic data
    df_inorganics = pandas.read_excel(inp['path_inorganics'])
    df_inorganics.apply(pandas.to_numeric, errors = 'ignore')
    df_inorganics.drop(columns = ['charge_number_Q1','metabolic_group'], inplace = True)
    # Select metabolites
    df_inorganics = df_inorganics[df_inorganics['compound_id'].isin(df_metabolites['compound_id'].unique())].copy()
    # Get priorization metabolites
    df_metabolites_literature = df_metabolites[df_metabolites['priorization']]
    # Mark metabolites in metabolite dataframe
    df_metabolites_hold = df_metabolites[
        (df_metabolites['compound_id'].isin(df_inorganics['compound_id']))&
        (~df_metabolites['compound_id'].isin(df_metabolites_literature['compound_id']))
        ].copy()
    # Set priorization
    for index, row in df_metabolites_hold.iterrows():
        df_metabolites.at[index, 'priorization'] = True
    return df_inorganics, df_metabolites

def development_parse_data_predictions(inp, df_metabolites):
    """
    Parse and prepare prediction data.

    Parse and preprocess prediction data for database creation.
    
    Parameters
    ----------
    inp : dict
        Input dictionary.
    df_metabolites : dataframe
        Preprocessed metabolite dataframe.

    Returns
    -------
    df_predictions : dataframe
        Dataframe with parsed prediction data.
    """
    # Select complementary metabolites
    df_metabolites_predict = df_metabolites.copy()  
    # Get all predictions in folder
    list_prediction_all = [item for item in os.listdir(inp['path_predictions']) if item.endswith('.txt')]
    # Select unique modes
    df_PosNeg = df_metabolites_predict[df_metabolites_predict['mode'].isin(['Pos','Neg'])].copy()
    # Select mode specific predictions
    list_prediction_select = [abb+'_'+mode+'.txt' for abb, mode in zip(df_PosNeg['compound_id'], df_PosNeg['mode']) if abb+'_'+mode+'.txt' in list_prediction_all]
    # Select both modes
    if 'Both' in df_metabolites_predict['mode'].unique():
        # Zwitterions without IBG1 data
        df_Both = df_metabolites_predict[(df_metabolites_predict['mode'] == 'Both')].copy()
        list_prediction_both = [
            abb+'_'+'Pos'+'.txt' 
            for abb in df_Both['compound_id'] 
            if abb+'_'+'Pos'+'.txt' in list_prediction_all]\
                +\
            [abb+'_'+'Neg'+'.txt' 
            for abb in df_Both['compound_id'] 
            if abb+'_'+'Neg'+'.txt' in list_prediction_all]
        # Append lists
        list_prediction_select = list_prediction_select + list_prediction_both
    # CEs from predictions that should be concidered 
    energy_name = ['energy0','energy1','energy2']
    energy_ce = [10,20,40]
    mapper_ce = dict(zip(energy_name, energy_ce))
    # Minimal threshold (relative intensity in %) for selection
    threshold_fragment = 5
    # Cycle predictions
    df_predictions = pandas.DataFrame()
    for prediction in list_prediction_select:
        abb, mode, filetype = re.split('_|\.', prediction)
        index = df_metabolites_predict[df_metabolites_predict['compound_id'] == abb].index[0]
        mw = df_metabolites_predict.at[index,'Molecular weight']
        charge = df_metabolites_predict.at[index,'Charge']
        name = df_metabolites_predict.at[index,'compound_name']
        prio = df_metabolites_predict.at[index,'priorization']
        # Calculate m/z
        if charge == 0:
            # For neutral substances, we account only single charges
            if mode == 'Pos':
                mz = mw + 1
            else:
                mz = mw - 1
        else:
            # Kegg may show MW of protonated or deprotonated substances (e.g. NAD+), therefore only divide by n_charges
            mz = mw/abs(charge)  
        file_prediction = inp['path_predictions'].joinpath(prediction)
        # Parse prediction
        try:
            list_tuple_results = development_predictionfile_to_predictionlist(file_prediction)
        except:
            print(prediction)

        # Select prediction
        if prio == True:
            indice = []
        else:
            ### Select only one prediction spectra (one CE) based on molecular size. 
            ### Results in less possible mass transitions to avoid isobarics compared to 3 fragment spectra.
            ### Still, smaller molecules respond better to small CE.
            indice = []
            if mw <= 100:
                indice.append(0)
            elif (mw > 100) & (mw <= 320):
                indice.append(1)
            else:
                indice.append(2)
        #indice = [0,1,2]

        # Create dataframe of prediction file
        for ind in indice:
            try:
                df_fragments = development_predictionlist_to_predictiondataframe(list_tuple_results[ind][1])
                # Drop q1 ion
                index_Q1 = df_fragments[df_fragments['Q3 [m/z]'] == mz].index
                df_fragments.drop(index_Q1 , inplace = True)
                # Get the product ions
                df_topN = df_fragments[df_fragments['relative intensity'] >= threshold_fragment].copy()
                # Add columns
                df_topN['compound_name'] = name
                df_topN['compound_id'] = abb
                df_topN['Q1 [m/z]'] = mz
                df_topN['mode'] = mode
                df_topN['CE [V]'] = mapper_ce[list_tuple_results[ind][0]]
                df_predictions = df_predictions.append(df_topN, sort = False)
            except:
                print(f'Prediction: Please check analyte {abb}_{mode}.')

    # Preallocate MSMS parameter
    df_predictions = df_predictions.drop(columns = 'relative intensity')
    df_predictions['DP [V]'] = None
    df_predictions['EP [V]'] = 10
    df_predictions['CXP [V]'] = 4
    df_predictions['dwelltime [ms]'] = 50
    df_predictions['reference'] = 'predicted'
    return df_predictions

def development_create_database(df_literature, df_inorganics, df_predictions, inp):
    """
    Create mass transition database.

    Create mass transition database based on literature, inorganic and prediction data.
    
    Parameters
    ----------
    df_literature : dataframe
        Dataframe with parsed literature data.
    df_inorganics : dataframe
        Dataframe with parsed inorganics data.
    df_predictions : dataframe
        Dataframe with parsed prediction data.
    inp : dict
        Input dictionary.

    Returns
    -------
    df_database : dataframe
        Dataframe with joined mass transitions.
    """
    # Preallocate database
    df_database = pandas.DataFrame()
    # Append database with input
    df_database = df_database.append(df_literature, sort = False)
    df_database = df_database.append(df_inorganics, sort = False)
    df_database = df_database.append(df_predictions, sort = False)
    # Fill database with standard parameter
    df_database['EP [V]'].fillna(value = 10, inplace = True)
    df_database['CXP [V]'].fillna(value = 4, inplace = True)
    df_database['DP [V]'].fillna(value = 0, inplace = True)
    df_database = df_database.reset_index(drop = True)
    # Regression for DP
    inp = development_create_database_linear_declustering_potential(df_literature, inp)
    DP_slope = inp['DP_slope']
    DP_intercept = inp['DP_intercept']
    # Cycle analytes
    for index, row in df_database.iterrows():
        # Set declustering potential
        if row['DP [V]'] == 0:
            df_database.at[index,'DP [V]'] = round(DP_slope*row['Q1 [m/z]'] + DP_intercept)
        elif row['DP [V]'] >= 200:
            df_database.at[index,'DP [V]'] = 200
        else:
            None
        # Set minimal and maximal collision energy
        if row['CE [V]'] < 5:
            df_database.at[index,'CE [V]'] = 5
        elif row['CE [V]'] > 130:
            df_database.at[index,'CE [V]'] = 130
        else:
            None
        # Set entrance potential
        if row['EP [V]'] > 15:
            df_database.at[index,'EP [V]'] = 15
        # Set collision cell exit potential
        if row['CXP [V]'] > 55:
            df_database.at[index, 'CXP [V]'] = 55

    # Prioritize mass transitions of inhouse and literature data before predictions 
    df_database = df_database.drop_duplicates(subset=['compound_id','mode','Q1 [m/z]','Q3 [m/z]'], keep = 'first')
    return df_database

def development_get_isobarics(df_database):
    """
    Get isobarics of database.

    Get isobaric mass transitions of joined database.
    
    Parameters
    ----------
    df_database : dataframe
        Dataframe with joined mass transitions.

    Returns
    -------
    df_isobarics : dataframe
        Dataframe with isobaric mass transitions.
    """
    # Create copy
    df_all = df_database.copy()
    # Preallocation
    list_isobarics = []
    # Cycle modes
    for mode in set(df_all['mode']):
        # Select mode dataframe
        df_mode = df_all.groupby(['mode']).get_group(mode)
        # Cycle all Q1
        for q1 in set(df_mode['Q1 [m/z]']):
            df_q1 = df_mode.groupby(['Q1 [m/z]']).get_group(q1)
            # Cycle all Q3
            for q3 in set(df_q1['Q3 [m/z]']):
                df_isobarics = df_q1.groupby(['Q3 [m/z]']).get_group(q3).drop_duplicates(subset=['compound_id'])
                # Append if isobarics present
                if len(df_isobarics) > 1:
                    list_isobarics.append([q1, q3, len(df_isobarics), mode, df_isobarics['compound_id']])
    # Preallocation
    df_isobarics = pandas.DataFrame()
    # Create report
    for report in list_isobarics:
        df_isobarics = df_isobarics.append(pandas.Series(report[0:4]), ignore_index = True)
    # If isobarics present
    if len(df_isobarics) != 0:
        df_isobarics.columns = ['Q1 [m/z]', 'Q3 [m/z]', 'number isobarics', 'mode']
        # Cycle isobarics
        for n in range(1, int(numpy.nanmax(df_isobarics['number isobarics']))+1):
            list_name = []
            for report in list_isobarics:
                df_names = report[4].reset_index()
                try:
                    list_name.append(df_names.iloc[n-1,1])
                except IndexError:
                    list_name.append(None)
            s_name = pandas.Series(list_name)
            df_isobarics.insert(3+n, f'Molecule_{n}', s_name)
    return df_isobarics

def development_extract_parameter(df_database, df_isobarics, inp):
    """
    Get isobaric ranking.

    Get isobaric ranking based on priorization.
    
    Parameters
    ----------
    df_database : dataframe
        Dataframe with joined mass transitions.
    df_isobarics : dataframe
        Dataframe with isobaric mass transitions.
    inp : dict
        Input dictionary.

    Returns
    -------
    df_mrms_all : dataframe
        Dataframe with joined and ranked mass transitions.
    """
    # Preallocate final mass transition dataframe
    df_mrms_all = pandas.DataFrame()
    df_database = df_database.sort_values(['origin']).copy()
    # Cycle inhouse data
    for abb in df_database['compound_id'].unique():
        # Get inhouse data
        df_abb = df_database.groupby(['compound_id']).get_group(abb).copy().reset_index(drop = True)
        # Cycle modes
        for mode in set(df_abb['mode']):
            df_mode = df_abb.groupby(['mode']).get_group(mode)
            df_checked = development_extract_parameter_check_for_isobarics(df_mode, df_isobarics)
            # Append
            df_mrms_all = df_mrms_all.append(df_checked, sort = False)
    # Reset index
    df_mrms_all = df_mrms_all.reset_index(drop = True)
    return df_mrms_all

def development_select_transitions(df_mrms_all, df_isobarics, inp):
    """
    Select mass transitions.

    Select mass transitions based on reference, convolution and isobaric ranking.
    
    Parameters
    ----------
    df_mrms_all : dataframe
        Dataframe with joined mass transitions.
    df_isobarics : dataframe
        Dataframe with isobaric mass transitions.
    inp : dict
        Input dictionary.

    Returns
    -------
    df_mrms_single : dataframe
        Dataframe with single mass transitions.
    df_isobarics_new : dataframe
        Dataframe with actual occuring isobarics.
    """
    # Preallocation
    df_mrms_select = pandas.DataFrame()
    list_isobaric_index = []
    # Cycle compounds
    for abb in df_mrms_all['compound_id'].unique():
        # Get compound
        df_abb = df_mrms_all.groupby(['compound_id']).get_group(abb).copy()
        if not df_isobarics.empty:
            df_iso = df_isobarics[df_isobarics.values == abb].copy()
        
        # If IBG data for zwitterions
        if (len(set(df_abb['mode']))>1) & ('IBG1' in set(df_abb['reference'])):
            df_abb = df_abb[df_abb['mode'].isin(set(df_abb[df_abb['reference']=='IBG1']['mode']))]
        # Cycle modes
        for mode in set(df_abb['mode']):
            df_mode = df_abb.groupby(['mode']).get_group(mode).copy()
            if not df_isobarics.empty:
                if mode in set(df_iso['mode']):
                    df_iso = df_iso.groupby(['mode']).get_group(mode).copy()
            if True in set(df_mode['isobarics avoided']):
                # Isobarics avoided
                df_hold = df_mode.groupby(['isobarics avoided']).get_group(True).copy().drop_duplicates(['compound_id'], keep = 'first')
            else:
                # Isobarics not avoided
                df_iso = df_iso[df_iso['number isobarics'] == min(df_iso['number isobarics'])].copy().drop_duplicates(['number isobarics'], keep = 'first')
                list_isobaric_index.append(df_iso.index[0])
                # Select isobaric, select highest ranking isobaric
                df_hold = df_mode[
                    (df_mode['Q1 [m/z]'].isin(df_iso['Q1 [m/z]'].unique())) & (df_mode['Q3 [m/z]'].isin(df_iso['Q3 [m/z]'].unique()))
                    ].copy().drop_duplicates(['compound_id'], keep = 'first')
            # Append corresponding transition
            df_mrms_select = df_mrms_select.append(df_hold, sort = False)
    # Select actual isobarics, drop duplicates to avoid multiple isobaric selection by narrow data
    df_isobarics_new = df_isobarics[df_isobarics.index.isin(list_isobaric_index)].copy()
    # Select mode for undecided mass transitions
    df_mrms_single, df_isobarics_append = development_select_transitions_mode(df_mrms_select, df_isobarics, inp)
    # Append isobarics
    df_isobarics_new = df_isobarics_new.append(df_isobarics_append, sort = False).dropna(how='all', axis=1)
    return df_mrms_single, df_isobarics_new

def development_select_transitions_mode(df_mrms_select, df_isobarics, inp):
    """
    Select undecided mass transitions.

    Select undecided mass transitions based on ionization mode.
    
    Parameters
    ----------
    df_mrms_select : dataframe
        Dataframe with undecided mass transitions.
    df_isobarics : dataframe
        Dataframe with isobaric mass transitions.
    inp : dict
        Input dictionary.

    Returns
    -------
    df_mrms_single : dataframe
        Dataframe with single mass transitions.
    df_isobarics_append : dataframe
        Dataframe with actual occuring isobarics after mode selection.
    """
    # Get non duplicates, single mode
    df_mrms_single = df_mrms_select[~df_mrms_select.duplicated(subset=['compound_id'], keep = False)].copy()
    # Get undecided mass transitions with both modes
    df_no_dec = df_mrms_select[df_mrms_select.duplicated(subset=['compound_id'], keep = False)].copy()
    # Cycle compound
    list_isobaric_index = []
    for abb in set(df_no_dec['compound_id']):
        df_abb = df_no_dec.groupby(['compound_id']).get_group(abb).reset_index(drop = True)
        if not df_isobarics.empty:
            df_iso = df_isobarics[df_isobarics.values == abb].copy()
        if len(set(df_abb['isobarics avoided'])) == 2:
            # If True and False in isobarics avoided, get both, True for single signal, False for False-Positives (Molecule is still there)
            df_hold = df_abb.copy()
            if not df_isobarics.empty:
                df_iso = df_iso[
                        df_iso['mode'].isin(set(df_hold['mode'])) & 
                        df_iso['Q3 [m/z]'].isin(set(df_hold['Q3 [m/z]']))
                        ]
                list_isobaric_index.append(df_iso.index[0])
        else:
            # Either both avoid or dont avoid isobarics
            if True in set(df_abb['isobarics avoided']):
                # Both avoid isobarics
                if ('predicted' in set(df_abb['reference'])) & (len(set(df_abb['reference']))>1):
                    # Priorize literature
                    df_hold = df_abb[~df_abb['reference'].isin(['predicted'])].copy()
                else:
                    # Both avoid isobarics, both predicted, get higher ranking fragment, 0 > 1 > 2...
                    if len(set(df_abb['ranked fragment'])) > 1:
                        # Different fragment quality
                        df_hold = df_abb[df_abb['ranked fragment'] == numpy.nanmin(df_abb['ranked fragment'])].copy()
                    else:
                        # Same fragment quality, append to smaller mode
                        if len(df_mrms_single[df_mrms_single['mode']=='Pos']) <= len(df_mrms_single[df_mrms_single['mode']=='Neg']):
                            df_hold = df_abb[df_abb['mode'] == 'Pos'].copy()
                        else:
                            df_hold = df_abb[df_abb['mode'] == 'Neg'].copy()
            else:
                # Both dont avoid isobarics, get both, False and False for False-Positives (Molecule is still there)
                df_hold = df_abb.copy()
                if not df_isobarics.empty:
                    df_iso = df_iso[
                            df_iso['mode'].isin(set(df_hold['mode'])) & 
                            df_iso['Q3 [m/z]'].isin(set(df_hold['Q3 [m/z]']))
                            ]
                    list_isobaric_index.append(df_iso.index[0])         
        # Append
        df_mrms_single = df_mrms_single.append(df_hold, sort = False)
    # Select actual isobarics, drop duplicates to avoid multiple isobaric selection by narrow data
    df_isobarics_append = df_isobarics[df_isobarics.index.isin(list_isobaric_index)].copy()
    # Sort values
    df_mrms_single = df_mrms_single.sort_values(['mode','Q1 [m/z]'], ascending = True).reset_index(drop = True)
    return df_mrms_single, df_isobarics_append

def development_group_convoluted_signals(df_mrms_single, df_isobarics_new, inp):
    """
    Group convoluted signals.

    Group convoluted mass transitions to avoid redundant signal acquisition.
    
    Parameters
    ----------
    df_mrms_single : dataframe
        Dataframe with single mass transitions.
    df_isobarics_new : dataframe
        Dataframe with actual occuring isobarics.
    inp : dict
        Input dictionary.

    Returns
    -------
    df_mrms_convoluted : dataframe
        Dataframe with final mass transitions.
    df_mrms_dropped : dataframe
        Dataframe with dropped mass transitions if selected.
    """
    # Preallocate dataframes
    df_mrms_convoluted = pandas.DataFrame()
    df_mrms_dropped = pandas.DataFrame()
    # If isobarics not avoided, group superimposed signals
    if False in set(df_mrms_single['isobarics avoided']):
        # Get isobarics
        df_isobarics_not_avoided = df_mrms_single.groupby(['isobarics avoided']).get_group(False).copy()
        # Append avoided mrms
        df_isobarics_avoided = df_mrms_single.drop(df_isobarics_not_avoided.index).copy()
        df_mrms_convoluted = df_mrms_convoluted.append(df_isobarics_avoided, sort = False)
        # Append convoluted mrms
        df_isobarics_not_avoided = df_isobarics_not_avoided.drop_duplicates(subset = ['mode','Q1 [m/z]','Q3 [m/z]'], keep = 'first').reset_index(drop = True)
        # Cycle isobarics
        for index, row in df_isobarics_not_avoided.iterrows():
            # Get information
            abb = row['compound_id']
            # Select mode of isobaric dataframe
            df_mode = df_isobarics_new.groupby(['mode']).get_group(row['mode'])
            # Select Q1 in mode of isobaric dataframe
            df_Q1 = df_mode.groupby(['Q1 [m/z]']).get_group(row['Q1 [m/z]'])
            # Select Q3 of Q1 in mode of isobaric dataframe
            df_Q3 = df_Q1.groupby(['Q3 [m/z]']).get_group(row['Q3 [m/z]']).copy().reset_index(drop = True).dropna(axis = 1)
            # Get abbreviations of isobarics
            list_abb = list(df_Q3.iloc[0,4:])
            sum_abb = '_'.join(list_abb)
            row['compound_id'] = sum_abb
            # Drop sumpeaks
            if inp['drop_sumpeaks'] == True:
                if numpy.nanmean(df_Q3['number isobarics']) < inp['drop_sumpeaks_no']:
                    df_mrms_convoluted = df_mrms_convoluted.append(row, sort = False)
                else:
                    df_mrms_dropped = df_mrms_dropped.append(row, sort = False)
            else:
                df_mrms_convoluted = df_mrms_convoluted.append(row, sort = False)
    
        df_mrms_convoluted = df_mrms_convoluted.reset_index(drop = True).sort_values(['mode','Q1 [m/z]'], ascending=True)
    else:
        df_mrms_convoluted = df_mrms_single.copy()   
    return df_mrms_convoluted, df_mrms_dropped

def development_format_output(df_mrms_convoluted, inp):
    """
    Format output.

    Format output for AB Sciex software.
    
    Parameters
    ----------
    df_mrms_convoluted : dataframe
        Dataframe with final mass transitions.
    inp : dict
        Input dictionary.

    Returns
    -------
    dict_methods : dictionary
        Dictionary with final DS-FIA-MS/MS methods.
    """
    # Preallocate
    dict_methods = {}
    # Set size of single method
    len_methods = 40
    # Set desired dwell time
    dwell_time = 50
    # Bring to ABSciex form
    df_methods = df_mrms_convoluted.drop(columns = ['compound_name','isobarics avoided','ranked fragment','reference']).copy()
    # Set column titles
    columns_sorting = ['Q1 [m/z]','Q3 [m/z]','dwelltime [ms]','compound_id','DP [V]','EP [V]','CE [V]','CXP [V]']
    # Set columns with potential
    columns_potential = ['DP [V]','EP [V]','CE [V]','CXP [V]']
    # Set uniform dwell time
    df_methods['dwelltime [ms]'] = dwell_time
    # Cycle modes
    for mode in set(df_methods['mode']):
        # Format mode dataframe
        df_mode = df_methods.groupby(['mode']).get_group(mode).copy()
        df_mode = df_mode.drop(columns = 'mode').reset_index(drop = True).filter(columns_sorting)
        # Set negative potentials for negative mode
        if mode == 'Neg':
            df_mode[columns_potential] = df_mode[columns_potential].apply(lambda x: -1*x)
        # Get number of methods for mode
        n_methods_mode = numpy.ceil((len(df_mode))/(len_methods))
        # Split dataframe into methods
        list_methods = list(numpy.array_split(df_mode, n_methods_mode))
        # Cycle methods and create dictionary
        for i, df in enumerate(list_methods, start = 1):
            dict_methods[f'{inp["path_method"].stem}_{mode}_{i}'] = df.reset_index(drop = True).copy()
    return dict_methods

def development_plot_resume(df, inp):
    """
    Plot resume.

    Plot resume of dsfiapy method development.
    
    Parameters
    ----------
    df : dataframe
        Dataframe with method information.
    inp : dict
        Input dictionary.
    """
    # Replace values
    list1 = ['IBG1','Inorganic']
    hold1 = df[df['reference'].isin(list1)]
    df.loc[hold1.index,'reference'] = 'inhouse'
    list2 = ['predicted','inhouse']
    hold2 = df[~df['reference'].isin(list2)]
    df.loc[hold2.index,'reference'] = 'literature'
    df['isobarics avoided'] = numpy.where(df['isobarics avoided'], 'unique', 'convolution')
    ### Plots ###
    variables = ['reference','mode','isobarics']
    fig, axes = plt.subplots(figsize = inp['figsize'], ncols = len(variables) , sharex = False, sharey = True)
    palette = create_palette(8, reverse = False)
    # Cycle variables
    for (i, variable), ax in zip(enumerate(variables), axes):
        # Preallocate bar lists
        bars_ref_1 = []
        bars_ref_2 = []
        bars_ref_3 = []
        # Values of each group
        if variable == 'reference':
            bars_ref_1.append(df['reference'].value_counts()['inhouse'])
            bars_ref_2.append(df['reference'].value_counts()['literature'])
            bars_ref_3.append(df['reference'].value_counts()['predicted'])
        if variable == 'mode':
            bars_ref_1.append(df['mode'].value_counts()['Pos'])
            bars_ref_2.append(df['mode'].value_counts()['Neg'])
        if variable == 'isobarics':
            bars_ref_1.append(df['isobarics avoided'].value_counts()['unique'])
            if 'convolution' in set(df['isobarics avoided']):
                bars_ref_2.append(df['isobarics avoided'].value_counts()['convolution'])
        # Heights of bar1 + bar2
        bars = numpy.add(bars_ref_1, bars_ref_2).tolist()
        # Names of group and bar width
        barWidth = 1
        # Values of each group
        if variable == 'reference':
            # Create bottom bars
            ax.bar(0, bars_ref_1, color = palette[0], edgecolor='white', label = 'inhouse')
            # Create middle bars, on top of the firs ones
            ax.bar(0, bars_ref_2, bottom = bars_ref_1, color=palette[1], edgecolor='white', label = 'literature')
            # Create top bars
            ax.bar(0, bars_ref_3, bottom = bars, color=palette[2], edgecolor='white', label = 'predicted')
        if variable == 'mode':
            # Create bottom bars
            ax.bar(0, bars_ref_1, color = palette[0], edgecolor='white', label = 'positive')
            # Create middle bars, on top of the firs ones
            ax.bar(0, bars_ref_2, bottom = bars_ref_1, color=palette[1], edgecolor='white', label = 'negative')
        if variable == 'isobarics':
            # Create bottom bars
            ax.bar(0, bars_ref_1, color = palette[0], edgecolor='white', label = 'unique')
            # Create middle bars, on top of the firs ones
            if 'convolution' in set(df['isobarics avoided']):
                ax.bar(0, bars_ref_2, bottom = bars_ref_1, color=palette[1], edgecolor='white', label = 'convolution')
        # Set x ticks
        ax.set_xticks([0])
        ax.set_xticklabels(labels = [variable], size = inp['labelsize'])
        # Set labels
        if ax.is_first_col:
            ax.set_ylabel('counts', size = inp['labelsize'], fontweight = 'bold')
        # Resize ticks
        ax.tick_params(axis='both', labelsize = inp['labelsize'])
        # Add legend
        ax.legend(bbox_to_anchor=(1.0, 1.0),loc="lower right", frameon = False, fontsize = inp['labelsize']-2)
        # Annotate bars
        for patch in ax.patches:
            # Get height of patch as label text
            value = patch.get_height()
            # Get x position of patch
            x_pos = patch.get_x() + patch.get_width() / 2.
            # Get y position of patch
            y_pos = patch.get_y() + patch.get_height() / 2.
            # Set label
            label = ax.annotate(value, (x_pos, y_pos), va = 'center', ha = 'center', fontsize = inp['labelsize'], fontweight = 'bold') 
    fig.tight_layout()
    fig.savefig(inp['path_development'].joinpath(f'summary_development.png'))
    fig.savefig(inp['path_development'].joinpath(f'summary_development.svg'), format = 'svg')
    return

def development_predictionfile_to_predictionlist(filepath):
    """
    Parse prediction csv file.

    Parse prediction csv file to list.
    
    Parameters
    ----------
    filepath : path
        Path to prediction.

    Returns
    -------
    list_results_energies : list
        List of tuple (Energy, Prediction)
    """
    # Open, read and close file
    file = open(filepath)
    list_results = file.read().splitlines()
    file.close()
    # Get collision energies in prediction file
    list_energies = [item for item in list_results if 'energy' in item]+[''] 
    # Preallocate list
    list_results_energies = []
    # Cycle collision energies
    for index, energy in enumerate([item for item in list_energies if item]):
        start = list_results.index(list_energies[index])
        end = list_results.index(list_energies[index+1])
        list_hold = list_results[start+1:end]
        list_results_energies.append((energy, list_hold))
    return list_results_energies
    
def development_predictionlist_to_predictiondataframe(
    list_prediction
    ):
    """
    Prediction as list to dataframe.

    Transform prediction in list to dataframe.
    
    Parameters
    ----------
    list_prediction : list
        Prediction in list format.

    Returns
    -------
    df : dataframe
        Prediction in dataframe
    """
    # List for sorting different data from one collision energy
    position_mz = 0
    position_intensity = 1
    # Preallocate list
    list_mz = []
    # Cycle predicted fragments
    for element in list_prediction:
        list_hold = element.split(' ')
        list_mz.append([round(float(list_hold[position_mz])), list_hold[position_intensity]])
    # Create dataframe
    df = pandas.DataFrame(list_mz, columns = ['Q3 [m/z]', 'relative intensity'])
    df = df.apply(pandas.to_numeric)
    df = df.sort_values(by = ['relative intensity'], ascending = False)
    df = df.reset_index(drop=True)
    return df
   
def development_extract_parameter_check_for_isobarics(
    df, df_isobarics
    ):
    """
    Check for isobarics.

    Check mass transitions for isobarics and rank fragments.
    
    Parameters
    ----------
    df : dataframe
        Dataframe to check for isobarics.
    df_isobarics : dataframe
        Dataframe with all isobarics.

    Returns
    -------
    df_new : dataframe
        Dataframe with isobaric information
    """
    df_new = df.copy()
    # Cycle rows of dataframe
    for index, row in df_new.iterrows():
        if len(df_isobarics) == 0:
            # Isobarics = False, avoided = True, number of isobarics = 0
            avoided = True
        else:
            # Isobarics = True
            # Preallocate list to store booleans for comparison of chosen ms parameters with isobarics
            list_bools = []
            # If mode in isobaric dataframe
            if row['mode'] in set(df_isobarics['mode']):
                # Get group of mode
                df_mode = df_isobarics.groupby(['mode']).get_group(row['mode'])
                # Check if q1 is in dataframe of mode
                bool_q1 = row['Q1 [m/z]'] in set(df_mode['Q1 [m/z]'])
                list_bools.append(bool_q1)
                if bool_q1 == False:
                    # q1 not in mode, isobarics, avoided = True, number of isobarics = index
                    # If mode and q1 not in isobarics mass transition can be used
                    avoided = True
                else:
                    # q1 in mode, isobarics
                    df_q1 = df_mode.groupby(['Q1 [m/z]']).get_group(row['Q1 [m/z]'])
                    # Get group of q1 in mode
                    bool_fragment = row['Q3 [m/z]'] in set(df_q1['Q3 [m/z]'])
                    list_bools.append(bool_fragment)
                    if list_bools == [True, False]:
                        # if mode and q1 but q3 not in isobarics, mass transition can be used
                        avoided = True
                    else:
                        avoided = False
            else:
                avoided = True
        df_new.at[index,'isobarics avoided'] = avoided
        df_new.at[index,'ranked fragment'] = index
    return df_new

def development_create_database_linear_declustering_potential(
    df_literature, inp
    ):
    """
    Get declustering potential.

    Model linear regression based on literature data and predict declustering potential.
    
    Parameters
    ----------
    df_literature : dataframe
        Dataframe with literature data for modeling.
    inp : dict
        Input dictionary.

    Returns
    -------
    inp : dict
        Input dictionary.
    """
    # If literature is provided, regress DP with Q1
    if not df_literature.empty:
        df_regress = df_literature.copy()
        df_regress = df_regress.drop_duplicates().dropna()
        x = numpy.array(df_regress['Q1 [m/z]'])
        y = numpy.array(df_regress['DP [V]'])
        coeffs = scipy.stats.linregress(x, y)
        inp['DP_slope'] = coeffs.slope
        inp['DP_intercept'] = coeffs.intercept
    else:
        inp['DP_slope'] = 0.11191370987567163
        inp['DP_intercept'] = 23.6217849957276
    return inp

def development_create_batch(
    path_allocation_batch, path_methods_batch, path_batch_folder,
    plate_code = '*96Greiner*', quality_control = False
    ):
    """
    Create measurement batch.

    Batch creation for DS-FIA-MS/MS method and provided sample allocation.
    Supports the following plate codes:
    96 Well MTP flat bottom:    '*96Greiner*'
    96 Well MTP V bottom:       '*96GreinerV*'
    
    Parameters
    ----------
    path_allocation_batch : str
        Raw string to allocation file.
    path_methods_batch : str
        Raw string to method dictionary in xlsx file.
    path_batch_folder : str
        Raw string to append for folder creation in AB Sciex Analyst software.
    plate_code : str
        Plate code of Analyst software, default '*96Greiner*'
    quality_control : bool
        Quality control used.

    Returns
    -------
    dict_batch : dict
        Dictionary with batch tables in xlsx format.
    """
    # Set parameter
    sample_id = None
    rack_code = 'Multi Drawer'
    rack_position = 1
    injection_volume = 5
    # Parse methods from method development
    dict_methods = pandas.read_excel(Path(path_methods_batch), sheet_name = None)
    # Parse allocation and create zipper
    df_sample_well = pandas.read_excel(path_allocation_batch)
    samples = [(i,j) for i,j in zip(df_sample_well['Sample Name'], df_sample_well['Well'])]

    # Get sample information for QC
    information_qc_reactor = pandas.Series([item[1] for item in df_sample_well['Sample Name'].str.split('_')]).unique()[0]
    information_qc_sample = pandas.Series([item[2] for item in df_sample_well['Sample Name'].str.split('_')]).unique()[0]
    information_qc_dil = pandas.Series([item[3] for item in df_sample_well['Sample Name'].str.split('_')]).unique()[0]
    information_qc_batch = pandas.Series([item[-1] for item in df_sample_well['Sample Name'].str.split('_')]).unique()[0]

    # Preallocation
    dict_batch = {}
    # Initiate sample count
    count_sample = 0
    count_plate = 2
    count_blank = 0
    count_qc = 0
    count_order = 0
    # Get number of necessary QC
    n_qc = int(numpy.ceil(len(samples)/6))
    if n_qc < 2:
        n_qc = 2
    qcs = [(f'QC_{information_qc_reactor}_{information_qc_sample}_{information_qc_dil}_{information_qc_batch}', 73+i) for i in range(n_qc)]
    blanks = [(f'Blank_{information_qc_reactor}_{information_qc_sample}_{information_qc_dil}_{information_qc_batch}', 85+i) for i in range(12)]
    # Cycle mode
    list_batch = []

    # Randomize samples
    numpy.random.shuffle(samples)

    for mode in ['Pos','Neg']:
        # Get methods of mode
        methods_mode = [item for item in dict_methods.keys() if mode in item]
        
        # Preallocate
        if quality_control == True:
            # Preallocate with QC
            array_sample = numpy.zeros(len(samples)+len(qcs), dtype=object)
            # Distribute QCs over sample array
            idx_qc = numpy.round(numpy.linspace(0, array_sample.shape[0] - 1, len(qcs))).astype(int)
            array_sample[idx_qc] = 'QC'
        else:
            # Preallocate without QC
            array_sample = numpy.zeros(len(samples), dtype=object)
        # Samples
        idx_sample = numpy.where(~(array_sample=='QC'))
        array_sample[idx_sample] = samples
        if quality_control  == True:
            array_sample[idx_qc] = qcs
        array_sample = array_sample.tolist()

        # Single blanks (no IDMS)
        single_blank_1, single_blank_2 = numpy.array_split(blanks, 2)
        single_blank_1 = single_blank_1.tolist()
        single_blank_2 = single_blank_2.tolist()

        ## TODO: double blanks (if IDMS is used, only IDMS)

        list_final = [(sample, int(well)) for sample, well in single_blank_1] + array_sample + [(sample, int(well)) for sample, well in single_blank_2]
        
        list_batch = []
        count_plate += 1
        for item in list_final:
            for method in methods_mode:
                count_order += 1
                if 'Blank' in item[0]:
                    count_blank +=1
                    counter = count_blank
                    sample_prefix = 'Blank'
                elif 'QC' in item[0]:
                    count_qc +=1
                    counter = count_qc
                    sample_prefix = 'QC'
                else:
                    count_sample +=1
                    counter = count_sample
                    sample_prefix = 'Sample'
                sample_id = count_order
                list_batch.append([item[0], sample_id, rack_code, rack_position, plate_code, count_plate, item[1], f'{method}',f'{path_batch_folder}\{sample_prefix}_{counter}', injection_volume])
        
        # Switch methods
        count_order += 1
        sample_id = count_order
        if mode == 'Pos':
            switch_name = 'Switch_PosNeg'
        else:
            switch_name = 'Switch_NegPos'
        list_batch.append(['Switch', sample_id, rack_code, rack_position, plate_code, count_plate, 85, switch_name, f'{path_batch_folder}\{switch_name}', injection_volume])

        df_batch = pandas.DataFrame(
                    list_batch, columns = [
                        'Sample Name', 
                        'Sample ID',
                        'Rack Code',
                        'Rack Position',
                        'Plate Code',
                        'Plate Position',
                        'Vial Position',
                        'Acquisition Method',
                        'Data File',
                        'Inj.Volume (µl)'
                    ]
        )
        # Create dictionary
        dict_batch[f'{mode}'] = df_batch.copy()

    # Save batch
    save_dict(dict_batch, Path(path_allocation_batch).parent, 'batch', single_files=False, index=False)

    # Create new allocation file
    if quality_control:
        df_qcs = pandas.DataFrame(qcs, columns = ['Sample Name','Well'])
        df_sample_well = df_sample_well.append(df_qcs)

    df_blanks = pandas.DataFrame(single_blank_1+single_blank_2, columns = ['Sample Name','Well'])
    df_sample_well = df_sample_well.append(df_blanks)
    df_sample_well['Well'] = df_sample_well['Well'].apply(lambda x: int(x))
    df_sample_well = df_sample_well.sort_values(['Well'], ascending = True).reset_index(drop = True)
    save_df(df_sample_well, Path(path_allocation_batch).parent, 'allocation_new', index = False)
    return dict_batch