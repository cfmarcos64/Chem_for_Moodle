# 🧪 Chem_To_Moodle Suite

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-orange)

A collection of three Streamlit applications designed to help chemistry teachers quickly generate Moodle-compatible questions using the **pmatchjme** plugin (students must draw the correct molecule in the JSME editor).

The three tools share the same philosophy: the teacher only provides compound names or SMILES strings, and the application handles SMILES lookup, image generation, JSME normalization, and XML export for Moodle.

## 🚀 Included Applications

1. **Molecule_To_Moodle**  
   Question type: "Draw the skeletal structure of [compound name]"

2. **Fischer_To_Moodle**  
   Question type: "Draw the skeletal structure corresponding to the given Fischer projection"

3. **Reaction_To_Moodle**  
   Question type: "Draw the missing molecule in the following reaction"

## ✨ Features

- Bilingual interface (Spanish default / English)
- Individual entry + bulk upload (CSV / Excel)
- Automatic SMILES lookup (NCI CIR primary + PubChem fallback)
- Sequential JSME normalization (single hidden editor instance)
- Base64-embedded images in exported XML
- Direct XML export compatible with pmatchjme question type
- Live preview of how the question will appear in Moodle

### Application-Specific Features
**Molecule_To_Moodle**

- No additional image generated
- Simple question text: "Draw the skeletal structure of X"

**Fischer_To_Moodle**

- Automatic generation of Fischer projection
- Rejects cyclic molecules with clear message
- Auto-converts common sugars to open-chain form
- Wavy bonds + warning for undefined stereocenters

**Reaction_To_Moodle**

- Input reactants, products, and agents over the arrow (SMILES or text)
- Quick name-to-SMILES search and field assignment
- Select which molecule is missing (reactant / product)
- Generates reaction diagram with missing part highlighted
- Flexible bulk format (R1,R2,… P1,P2,… A1,A2,… Missing_Name)

## 🛠️ How to Run the Suite

There are two primary ways to access and use this tool:

### Option 1: Use the Public Web Application (Recommended)

The application is deployed publicly and can be accessed directly through this link:

👉 https://create-pattern-match-with-molecular-editor-questions-for-moodle.streamlit.app/

### Option 2: Run Locally (Requires Python and Node.js)

To run the application in local development mode, you must run two processes simultaneously in separate terminals: the Streamlit server (Python) and the frontend component development server (Node/npm).

1. Clone the Repository and Install Python Dependencies:

git clone https://github.com/cfmarcos64/Create-pattern-match-with-molecular-editor-questions-for-Moodle

cd [repository-name]

// Install Python dependencies (skip if already done)
pip install -r requirements.txt


2. Run the Component Frontend (TERMINAL 1):

This step starts the component development server on http://localhost:3001. This is necessary for Streamlit to connect to the React component and see live changes.

// Navigate to the frontend directory
cd my_component/frontend
// Install JavaScript dependencies (only the first time)
npm install
// Start the component development server
npm run start


**Note:** Keep this terminal open and running while using the Streamlit application.

3. Run the Streamlit Application (TERMINAL 2):

Open a second terminal. Navigate back to the project root folder and run the application.

// Go back to the root directory
cd ../..
// Execute the main Streamlit application
streamlit run MoleculeToMoodleJSME.py

The Streamlit server will automatically connect to the component development server (Terminal 1).

📦 Requirements (requirements.txt)

streamlit>=1.31.0
rdkit>=2023.09.1
pandas>=2.0
openpyxl>=3.1
requests>=2.28
pubchempy>=0.4
pillow>=10.0
numpy>=1.23
matplotlib>=3.7

## 📁 File Structure

The repository is organized into two main parts: the Streamlit Python applications and the custom component frontend. The project directory should contain at least these files:

Chem_To_Moodle/
├── MoleculeToMoodleJSME.py
├── Fischer_for_Moodle.py
├── ReactionToMoodleJSME.py
├── my_component/                # Custom JSME Streamlit component
│   ├── __init__.py
│   └── frontend/                # React + Vite source
├── requirements.txt
├── README.md
└── LICENSE

## License

This project is licensed under the CC BY_NC_SA 4.0 License. See the [LICENSE](LICENSE) file for details.

*This tool was created to assist educators and chemists in quickly generating high-quality Moodle quiz content.*

## Contact
Created by Carlos Fernández Marcos
GitHub: @cfmarcos64
Email: cfmarcos@unex.es
Last updated: February 2026