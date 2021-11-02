import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime

class simulation:
    
    def __init__(self, 
                 pop_df, 
                 log_reg_params, 
                 life_tables,
                 save_xl_fname=None, 
                 save_np_fname=None, 
                 end_year=2068, 
                 start_year=2018):
        self.pop_df = pop_df
        self.life_tables = life_tables
        self.end_year=end_year
        self.start_year=start_year
        self.log_reg_params = log_reg_params
        self.save_xl_fname = save_xl_fname
        self.save_np_fname = save_np_fname
        self.output_columns = [
            "year", "race", "poverty", "never smoker", "former smoker", "non-methol smoker", "menthol smoker", "e-cig user", "dead"
        ]
        # TODO
        self.input_columns = [
            "blah"
        ]
        self.output_shape = ((self.end_year - self.start_year) * 4, len(self.output_columns))
        self.output = []
        
        return
    
    def simulate():
        """
        Calling this function causes 1 run of the simulation to happen.
        Results are written according to save_xl_fname and save_np_fname

        Args:
            None
        
        Output:
            self.output: the data written out from the simulation
        """
        now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        self.pop_arr = self.pop_df.to_numpy()
        self.output = np.zeros(self.output_shape)
        
        for cy in range(self.start_year, self.end_year):
            # update population
            update()
        
        # writeout the results of the simulation
        writeout(now_str)

        return self.output
        
    
    def update():
        """
        Update the smoking status of each person according to transition probabilities
        """
        # people will die according to life tables 

        # get transition probabilities and use them with relavant state vector
        transition_probs = self.softmax(np.matmul(self.pop_arr[:,:5], self.log_reg_params), 1)
        # sample the transition probs to determine next state

        # increment age and agegroup if needed

        # write 4 lines in self.output
        
        return 
    
    def writeout(now_str):
        """
        blah
        """
        
        if self.save_xl_fname:
            pd.DataFrame(self.output, columns=self.output_columns).to_excel(self.save_xl_fname + "_" + now_str)
        
        if self.save_np_fname:
            np.save(self.save_np_fname, self.output)

        return
    
    def softmax(a, ax):
        return np.exp(a)/np.sum(np.exp(a), axis=ax)