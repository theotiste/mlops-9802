import mlflow
import argparse

def main(recherche_modele, optimisation_modele):
    with mlflow.start_run():
        # Enregistrement des paramètres
        mlflow.log_param("recherche_modele", recherche_modele)
        mlflow.log_param("optimisation_modele", optimisation_modele)
        
        # Enregistrement d'une métrique fictive
        mlflow.log_metric("score", 0.85)
        
        print(f"Recherche modèle : {recherche_modele}, Optimisation modèle : {optimisation_modele}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recherche_modele", type=str, default="default_value")
    parser.add_argument("--optimisation_modele", type=str, default="default_value")
    args = parser.parse_args()
    
    main(args.recherche_modele, args.optimisation_modele)
