from simulation import Simulation
import pandas as pd
import numpy as np
import argparse
import os

def main(args):
    # 0 = male
    # 1 = female

    # Get life tables
    life_table_dict = {}
    life_table_dict[2016] = {}
    life_table_dict[2017] = {}
    life_table_dict[2018] = {}
    life_table_dict[2016][0] = pd.read_excel(os.path.join("..","..","life_tables","2016","Males","life_table_2016_male.xlsx")).to_numpy()[2:-1,1]
    life_table_dict[2017][0] = pd.read_excel(os.path.join("..","..","life_tables","2017","Males","life_table_2017_male.xlsx")).to_numpy()[2:-1,1]
    life_table_dict[2018][0] = pd.read_excel(os.path.join("..","..","life_tables","2018","Males","life_table_2018_male.xlsx")).to_numpy()[2:-1,1]
    life_table_dict[2016][1] = pd.read_excel(os.path.join("..","..","life_tables","2016","Females","life_table_2016_female.xlsx")).to_numpy()[2:-1,1]
    life_table_dict[2017][1] = pd.read_excel(os.path.join("..","..","life_tables","2017","Females","life_table_2017_female.xlsx")).to_numpy()[2:-1,1]
    life_table_dict[2018][1] = pd.read_excel(os.path.join("..","..","life_tables","2018","Females","life_table_2018_female.xlsx")).to_numpy()[2:-1,1]

    # Get smoking prevalences by age and sex
    smoking_prevalence_dict = {}
    smoking_prevalence_dict[2016] = {}
    smoking_prevalence_dict[2017] = {}
    smoking_prevalence_dict[2018] = {}
    smoking_prevalence_dict[2016][0] = pd.read_excel(os.path.join("..","..","smoking_prevalences_Feb9","Smoker_percentage16_M.xlsx")).to_numpy()[:,4::3]
    smoking_prevalence_dict[2017][0] = pd.read_excel(os.path.join("..","..","smoking_prevalences_Feb9","Smoker_percentage17_M.xlsx")).to_numpy()[:,4::3]
    smoking_prevalence_dict[2018][0] = pd.read_excel(os.path.join("..","..","smoking_prevalences_Feb9","Smoker_percentage18_M.xlsx")).to_numpy()[:,4::3]
    smoking_prevalence_dict[2016][1] = pd.read_excel(os.path.join("..","..","smoking_prevalences_Feb9","Smoker_percentage16_F.xlsx")).to_numpy()[:,4::3]
    smoking_prevalence_dict[2017][1] = pd.read_excel(os.path.join("..","..","smoking_prevalences_Feb9","Smoker_percentage17_F.xlsx")).to_numpy()[:,4::3]
    smoking_prevalence_dict[2018][1] = pd.read_excel(os.path.join("..","..","smoking_prevalences_Feb9","Smoker_percentage18_F.xlsx")).to_numpy()[:,4::3]

    # Get Releative Risks for current smokers vs nonsmoker and former smoker vs current smoker
    # According to a published review
    csvnsRR = pd.read_excel(os.path.join("..", "..", "smoking_prevalence", "current_smoker_mortality_vs_nonsmoker.xlsx")).to_numpy()[:,1:]
    fsvcsRR = pd.read_excel(os.path.join("..", "..", "smoking_prevalence", "former_smoker_mortality_vs_current_smoker.xlsx")).to_numpy()[:,1:]

    # Get population data
    pop_file_name = os.path.join("..","..","population_files_Feb8","population_file_sent_Feb8.xlsx")
    pop_df = pd.read_excel(pop_file_name)

    # Get logistic regression betas
    beta2345_f = os.path.join("..","..","Output_SM","Betas","Beta_Estimates_2345.xlsx")
    beta1_f = os.path.join("..","..","Output_SM","Betas","Beta_Estimates_1.xlsx")
    beta2345_arr = pd.read_excel(beta2345_f).to_numpy()[:,2:]
    beta1_arr = pd.read_excel(beta1_f).to_numpy()[:,2:]

    for i in range(args.number_replications):
        s = Simulation(pop_df=pop_df, 
                    beta2345=beta2345_arr, 
                    beta1=beta1_arr, 
                    life_tables=life_table_dict,
                    smoking_prevalences=smoking_prevalence_dict,
                    current_smoker_RR=csvnsRR,
                    former_smoker_RR=fsvcsRR,
                    save_xl_fname='xl_output',
                    save_np_fname='np_output',
                    save_transition_np_fname='transitions',
                    )

        s.simulate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify simulation parameters')
    parser.add_argument('number_replications', 
                        type=int,
                        default=1,
                        help='the number of relplications to do')
    main(parser.parse_args())