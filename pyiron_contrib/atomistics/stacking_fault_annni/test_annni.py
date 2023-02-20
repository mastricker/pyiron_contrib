import unittest

from pyiron_atomistics import Project
import numpy as np
from stacking_fault_annni import *


class TestANNNI(unittest.TestCase):
    
    def setUp(self):
        self.project_name = "annni_test_project"
        self.potential = "2009--Kim-Y-M--Mg-Al--LAMMPS--ipr1"
        self.temp_max = 200
        self.temp_steps = 5
        self.job_name = "test_anni"
        self.element = "Mg"
        self.lattice_constant = 3.21 # Å

        self.results = np.array([[0.9032931406387965,
                                  0.9019084480333819,
                                  0.8697414163671285,
                                  0.8577833951545687,
                                  0.8443928293181279],
                                 [1.7979261580527426,
                                  1.7949775833435049,
                                  1.7307336765191987,
                                  1.706389455046523,
                                  1.6791711748327744]])


    def test_annni_job(self):
        project = Project(self.project_name)
        job = project.create_job(job_type=StackingFaultANNNI,
                                 job_name=self.job_name,
                                 delete_existing_job=True)
        
        job.reference_structure = project.create.structure.bulk(self.element,
                                                                a=self.lattice_constant,
                                                                orthorhombic=False).repeat(2)
        job.potential = self.potential 
        job.calc_anni(temperature=self.temp_max, steps=self.temp_steps)
        job.run()

        result = job.output["stacking_fault_energy"]

        sfI1 = result[0]
        sfI2 = result[1]

        ref_sfI1 = self.results[0]
        ref_sfI2 = self.results[1]

        for val, ref_val in zip(sfI1, ref_sfI1):
            self.assertLessEqual(np.abs(ref_val-val), 1e-8)

        for val, ref_val in zip(sfI2, ref_sfI2):
            self.assertLessEqual(np.abs(ref_val-val), 1e-8)

        
