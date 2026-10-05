"""Point d'entrée unique vers les LLM : Mistral (API), Groq (API) ou Ollama (local).
Le reste du code appelle discuter() sans savoir quel fournisseur répond."""
import logging
import os
import time

import config

logger = logging.getLogger(__name__)

_clients = {}


def _cle(nom_variable):
    """Les clés API sont lues dans les variables d'environnement, jamais dans le code.
    Sous Windows, si la variable n'a pas été transmise au processus (un client MCP ne
    transmet qu'une liste restreinte de variables au serveur qu'il lance), on la lit
    directement dans les variables utilisateur de Windows."""
    cle = os.environ.get(nom_variable)
    if not cle and os.name == "nt":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as registre:
                cle = winreg.QueryValueEx(registre, nom_variable)[0]
        except OSError:
            cle = None
    if not cle:
        raise RuntimeError(f"Variable d'environnement {nom_variable} absente (voir le README).")
    return cle


def _client(fournisseur):
    if fournisseur not in _clients:
        if fournisseur == "mistral":
            from mistralai.client import Mistral
            _clients[fournisseur] = Mistral(api_key=_cle("MISTRAL_API_KEY"))
        elif fournisseur == "groq":
            from groq import Groq
            # Les nouvelles tentatives sont gérées par discuter(), pareil pour tous les fournisseurs
            _clients[fournisseur] = Groq(api_key=_cle("GROQ_API_KEY"), max_retries=0)
        else:
            raise ValueError(f"Fournisseur inconnu : {fournisseur}")
    return _clients[fournisseur]


def _appeler(messages, fournisseur, modele, json_strict):
    if fournisseur == "ollama":
        import ollama
        resultat = ollama.chat(
            model=modele, messages=messages,
            format="json" if json_strict else "", options={"temperature": 0},
        )
        return resultat["message"]["content"]

    format_reponse = {"type": "json_object"} if json_strict else None
    if fournisseur == "mistral":
        resultat = _client("mistral").chat.complete(
            model=modele, messages=messages, temperature=0, response_format=format_reponse)
    else:
        resultat = _client(fournisseur).chat.completions.create(
            model=modele, messages=messages, temperature=0, response_format=format_reponse)
    return resultat.choices[0].message.content


def discuter(messages, fournisseur, modele, json_strict=False):
    """Envoie une conversation au LLM (température 0) et retourne le texte de sa réponse.
    Si l'API répond "trop de requêtes" (429) ou est momentanément indisponible (5xx),
    on attend de plus en plus longtemps (2, 4, 8... s) avant de réessayer."""
    for tentative in range(config.LLM_ESSAIS):
        try:
            return _appeler(messages, fournisseur, modele, json_strict).strip()
        except Exception as erreur:
            code = getattr(erreur, "status_code", None)
            temporaire = code == 429 or (code is not None and code >= 500)
            if not temporaire or tentative == config.LLM_ESSAIS - 1:
                raise
            attente = 2 ** (tentative + 1)
            logger.warning(f"{fournisseur}/{modele} : erreur {code}, nouvel essai dans {attente} s")
            time.sleep(attente)
