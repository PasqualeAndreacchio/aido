import os
from typing import Dict

import torch
from calo_opt.interface_simple import AIDOUserInterfaceExample  # Import your derived class
from calo_opt.plotting import CaloOptPlotting

import aido


class UIFullCalorimeter(AIDOUserInterfaceExample):

    @classmethod
    def constraints(
            self,
            parameter_dict: aido.SimulationParameterDictionary,
            parameter_dict_as_tensor: Dict[str, torch.Tensor]
            ) -> torch.Tensor:
        """ Additional constraints to add to the Loss.

        Use the Tensors found in 'parameter_dict_as_tensor' (Dict) to compute the constraints
        and return a 1-dimensional Tensor. Note that missing gradients at this stage will
        negatively impact the training of the optimizer.

        Use the usual 'parameter_dict' instance to access additional information such as
        boundaries, costs per item and all other stored values.

        In this example, we add the cost per layer for all six layers by looping over the
        index of the layer (0, 1, 2) and their type (absorber / scintillator). Using the

        """

        detector_length_list = []
        cost_list = []

        for i in range(3):
            for name in ["absorber", "scintillator"]:
                material_probabilities = parameter_dict_as_tensor[f"material_{name}_{i}"]
                material_cost = torch.tensor(
                    parameter_dict[f"material_{name}_{i}"].cost,
                    device=material_probabilities.device
                )
                layer_weighted_cost = material_probabilities * material_cost
                layer_thickness = parameter_dict_as_tensor[f"thickness_{name}_{i}"]

                cost_list.append(layer_thickness * layer_weighted_cost)
                detector_length_list.append(layer_thickness)

        max_length = parameter_dict["max_length"].current_value
        max_cost = parameter_dict["max_cost"].current_value
        detector_length = torch.stack(detector_length_list).sum()
        cost = torch.stack(cost_list).sum()
        detector_length_penalty = torch.mean(torch.nn.functional.relu((detector_length - max_length) / max_length)**2)
        max_cost_penalty = torch.mean(torch.nn.functional.relu((cost - max_cost) / max_cost)**2)
        return detector_length_penalty + max_cost_penalty

    def plot(self, parameter_dict: aido.SimulationParameterDictionary) -> None:
        plotter = CaloOptPlotting(self.results_dir)
        plotter.plot()
        return None


if __name__ == "__main__":

    min_value = 0.0

    parameters = aido.SimulationParameterDictionary([
        aido.SimulationParameter("thickness_absorber_0", 9.0, min_value=min_value),
        aido.SimulationParameter("thickness_scintillator_0", 37.0, min_value=min_value),
        aido.SimulationParameter(
            "material_absorber_0",
            "G4_Fe",
            discrete_values=["G4_Pb", "G4_Fe"],
            cost=[25, 4.166],
            probabilities=[0.10, 0.90]
        ),
        aido.SimulationParameter(
            "material_scintillator_0",
            "G4_POLYSTYRENE",
            discrete_values=["G4_PbWO4", "G4_POLYSTYRENE"],
            cost=[2500.0, 0.01],
            probabilities=[0.90, 0.10]
        ),
        aido.SimulationParameter("thickness_absorber_1", 10.0, min_value=min_value),
        aido.SimulationParameter("thickness_scintillator_1", 29.0, min_value=min_value),
        aido.SimulationParameter(
            "material_absorber_1",
            "G4_Fe",
            discrete_values=["G4_Pb", "G4_Fe"],
            cost=[25, 4.166],
            probabilities=[0.90, 0.10]
        ),
        aido.SimulationParameter(
            "material_scintillator_1",
            "G4_POLYSTYRENE",
            discrete_values=["G4_PbWO4", "G4_POLYSTYRENE"],
            cost=[2500.0, 0.01],
            probabilities=[0.10, 0.90]
        ),
        aido.SimulationParameter("thickness_absorber_2", 36.0, min_value=min_value),
        aido.SimulationParameter("thickness_scintillator_2", 27.5, min_value=min_value),
        aido.SimulationParameter(
            "material_absorber_2",
            "G4_Fe",
            discrete_values=["G4_Pb", "G4_Fe"],
            cost=[25, 4.166],
            probabilities=[0.10, 0.90]
        ),
        aido.SimulationParameter(
            "material_scintillator_2",
            "G4_POLYSTYRENE",
            discrete_values=["G4_PbWO4", "G4_POLYSTYRENE"],
            cost=[2500.0, 0.01],
            probabilities=[0.10, 0.90]
        ),
        aido.SimulationParameter("num_events", 800, optimizable=False),
        aido.SimulationParameter("max_length", 150, optimizable=False),
        aido.SimulationParameter("max_cost", 200_000, optimizable=False),
        aido.SimulationParameter("full_calorimeter", True, optimizable=False)
    ])

    results_dir: str = "First_Calo_2107"
    os.system(f"rm -f /mnt/work/aido/Results/{results_dir}")
    aido.optimize(
                parameters=parameters,
                user_interface=UIFullCalorimeter,
                simulation_tasks=25,
                max_iterations=60,
                threads=32,
                results_dir=f"/mnt/work/aido/Results/{results_dir}",
                description="""
                Optimization of a sampling calorimeter with cost and length constraints.
                Includes the optimization of discrete parameters, specific plotting functions
                """
    )
    # Set the correct result directory
    #os.system(f"mv /mnt/work/aido/examples/{results_dir}/task_outputs /mnt/work/aido/Results/{results_dir}")
    os.system(f"rm -fr /mnt/work/aido/examples/{results_dir}")

    ui = UIFullCalorimeter()
    ui.results_dir = f"/mnt/work/aido/Results/{results_dir}"
    ui.plot(parameters)
