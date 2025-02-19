import mlflow
import argparse

def main(recherche_modele, optimisation_modele):
    with mlflow.start_run():
        # Enregistrement des paramètres
        mlflow.log_param("recherche_modele", recherche_modele)
        mlflow.log_param("optimisation_modele", optimisation_modele)

        import pandas as pd
        import os

        def save_metrics_to_csv(experiment_name, run_id, params, metrics):
            """Sauvegarde les métriques dans un CSV."""
            results_dir = "mlflow_results"
            os.makedirs(results_dir, exist_ok=True)

            file_path = os.path.join(results_dir, f"{experiment_name}_metrics.csv")

            # Construire un dataframe
            df = pd.DataFrame([{**params, **metrics, "run_id": run_id}])

            # Sauvegarde en CSV (ajout si fichier existe déjà)
            df.to_csv(file_path, mode="a", index=False, header=not os.path.exists(file_path))
            print(f"📁 Metrics saved in {file_path}")

            # Exemple de métriques enregistrées
            params = {"recherche_modele": recherche_modele, "optimisation_modele": optimisation_modele}
            metrics = {"accuracy": 0.85, "loss": 0.15}

            save_metrics_to_csv("recherche_modele", mlflow.active_run().info.run_id, params, metrics)
  
        
            # Enregistrement d'une métrique fictive
            mlflow.log_metric("score", 0.85)
        
            print(f"Recherche modèle : {recherche_modele}, Optimisation modèle : {optimisation_modele}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recherche_modele", type=str, default="default_value")
    parser.add_argument("--optimisation_modele", type=str, default="default_value")
    args = parser.parse_args()
    
    main(args.recherche_modele, args.optimisation_modele)
