# Lunar-Logic_SMUBIA
# Criminal Intelligence System: Advanced Crime Network Analysis

## Project Overview

This Criminal Intelligence System is an advanced data analytics solution designed to transform how investigative services analyze and understand complex criminal networks. By leveraging cutting-edge machine learning, natural language processing, and network analysis techniques, the system provides comprehensive insights into criminal relationships, patterns, and potential threats.

## Key Components

### 1. Data Preprocessing (`1 Data Preprocessing.ipynb`)
- Utilizes BERT and SpaCy for advanced text analysis
- Extracts structured entities from unstructured crime-related documents
- Identifies and categorizes:
  * Locations
  * People
  * Organizations
  * Crime Types

### 2. Data Cleaning (`2 Data Postprocessing.ipynb`)
- Standardizes location and person names
- Removes duplicate entries
- Ensures data consistency and quality
- Prepares data for advanced analysis

### 3. Relationship Extraction (`3 Row Relationships.ipynb`)
- Constructs complex relationship networks between entities
- Identifies connections such as:
  * Organizational Partnerships
  * Funding Relationships
  * Potential Criminal Collaborations
- Assesses evidence strength of relationships

### 4. Network Analysis (`4 Relationships Tablaeu.ipynb`)
- Builds a comprehensive crime network graph
- Calculates network metrics:
  * Degree Centrality
  * Betweenness Centrality
  * Clustering Coefficient
- Identifies key entities and crime patterns

### 5. Predictive Modeling (`5 Prediction.ipynb`)
- Develops machine learning model to predict repeat offenders
- Generates risk scores for entities
- Provides risk level classifications:
  * Very Low
  * Low
  * Medium
  * High
  * Very High

## Technologies Used

- Python
- Pandas
- SpaCy
- NetworkX
- BERT
- XGBoost
- Scikit-learn
- Tableau

## Key Features

1. **Advanced Entity Extraction**: Automatically identifies and categorizes entities from text
2. **Network Visualization**: Creates comprehensive crime network graphs
3. **Predictive Risk Scoring**: Assesses likelihood of repeat criminal activities
4. **Multi-dimensional Analysis**: Examines relationships across organizations and individuals

## How It Works

1. Ingest raw crime-related documents
2. Extract and standardize entities
3. Build relationship networks
4. Apply machine learning to predict risks
5. Generate actionable insights for investigators

## Example Outputs

- Relationship CSV files showing connections between entities
- Network graph visualizations
- Entity risk score rankings
- Detailed crime pattern analysis

## Potential Applications

- Law Enforcement Intelligence
- Organized Crime Investigation
- Proactive Threat Assessment
- Complex Network Relationship Mapping

## Installation

1. Clone the repository
