from CADETProcess.optimization import OptimizerBase


from ax.service.ax_client import AxClient, ObjectiveProperties
import numpy as np

class AxInterface(OptimizerBase):
    def __init__(self):
        self.ax_client = AxClient()
    
    def translate_variables(self, optimization_problem):
        return [{"name": var, "type": "range", "bounds": [0.0, 1.0]}for var in optimization_problem.independent_variable_names]


    

    def run(self, optimization_problem, name='unnamed', tol=1.0e-4, limit = 10, improve_limit=10, *args, **kwargs):

        objectives = dict([[obj,ObjectiveProperties(minimize=True)] for obj in optimization_problem.objective_labels])

        self.ax_client.create_experiment(
            name = name,
            parameters = self.translate_variables(optimization_problem),
            objectives=objectives
        )    
        def objective_function(x):
            eval =  optimization_problem.evaluate_objectives(np.array(list(x.values())), untransform=True)
            labels = [obj for obj in optimization_problem.objective_labels]
            return dict(zip(labels, [(x,0.0) for x in eval]))
        

        i = 0
        curr_min =  np.ones(len(objectives))
        unchanged = 0 
        while i <= limit and np.linalg.norm(curr_min, ord=np.inf) >= tol and unchanged < improve_limit:
            parameters, trial_index = self.ax_client.get_next_trial()
            obj_val = objective_function(parameters)
            x = np.array([obj[0] for obj in obj_val.values()])
            if np.linalg.norm(x, ord=np.inf) < np.linalg.norm(curr_min, ord=np.inf):
                curr_min = x
                unchanged = 0
            else:
                unchanged += 1
            self.ax_client.complete_trial(trial_index=trial_index, raw_data=obj_val)
            i += 1


        self.results.exit_flag = 0

        if len(objectives) == 1:
            best_parameters, values = self.ax_client.get_best_parameters()
            x = np.array(list(best_parameters.values()))
            self.run_post_evaluation_processing(x, curr_min, None, None, 1)
        else:
            best_parameters = list(self.ax_client.get_pareto_optimal_parameters().values())[0]
            values = np.array(list(best_parameters[1][0].values()))
            x = np.array(list(best_parameters[0].values()))
            self.run_post_evaluation_processing(x, values, None, None, 1)
        
        
        return self.results


class ServiceWrapper(AxInterface):
    def __str__(self):
        return 'ServiceAPI'