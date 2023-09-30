# Features

Falcondale aims to make Quantum Machine Learning easy to use but also highly adaptable to those with specific needs or willing to test different options. At current version it provides:

* Simple and easy **data preprocessing** for the purpose. You can bring your own data engineering practices but we also want to make it easy for those new to the field
* **Quantum Feature Selection**, including D-Wave job submission, using _state-of-the-art_ techniques
* **Quantum Classification** models with a diverse set of options: Quantum Support Vector Machines, Quantum Neural Networks, Variational Quantum Classifiers,...
* **Quantum Clustering** by means of quantum inspired and fully-quantum approaches like Quantum Approximate Optimization Algorithm (QAOA)

All these features count with a broad set of options so that each one can be customized and enable benchmarking. It also brings together some industry widely adopted frameworks so that contributions to these can be included as the usage grows.

* IBM's [Qiskit framework](https://qiskit.org/)
* Xanadu's [Pennylane framework](https://pennylane.ai/) for quantum differential programming
* D-Wave's [Ocean SDK](https://docs.ocean.dwavesys.com/en/stable/)

Future releases will also count with broader integrations so that users can select the target device in which their algorithms would run. target devices such as:

* D-Wave Cloud services
* Oxford Quantum Computing QCaaS
* Toshiba's SQBM+
* IBM Quantum Runtime supported devices
* QuEra's neutral atom devices
* _and more to be added_

This is a collaborative effort so we hope a growing list of integrations can be brought to Falcondale with the help of everyone!