# Round 2 Local Runbook

## Start the full local stack

1. Start the verified model API from the repository root:

   ```powershell
   .\.venv-model\Scripts\python.exe -m uvicorn model_api:app --host 127.0.0.1 --port 8000
   ```

2. Start Node in a second terminal:

   ```powershell
   cd round2-node
   npm start
   ```

3. Open `http://127.0.0.1:3000`. Mission Control uses same-origin API calls automatically; no backend URL entry is needed. Node calls the model at `http://127.0.0.1:8000` by default. Set `MODEL_URL` in the ignored `round2-node/.env` only to intentionally use a different model service.

The Python API loads `model_artefact_verified.joblib` on startup and exposes `/health`, `/predict`, and `/predict/batch`. It returns the EXP_C hybrid model result at the selected 0.50 threshold. Node exposes the operator API and serves the frontend.

## Email settings

SMTP host, port, TLS mode, username, sender, and app password are loaded from ignored `round2-node/.env`. Keep `SMTP_PASS` only in that file. Enter the recipient and enable live alerts in Mission Control Settings; these two settings remain in backend memory until Node restarts. Verify SMTP before relying on live alerts. The test-email button sends a real test message; an SMTP accepted status is not proof of inbox delivery.

Simulation uses random rows from `test.csv` and is marked as test data, not live telemetry. When live alerts are enabled, a recipient is saved, and SMTP credentials are configured, simulation automatically uses the saved recipient. Otherwise the run is dry-run. The default per-run email cap is one, with a hard cap of three.

## Current MVP boundaries

Implemented: model inference, bounded random simulation, local JSON event history, SMTP test and fault-alert delivery, and Mission Control status/results.

Not implemented: GraphQL, actual crew roster/specialist dispatch, persistent vector memory, repair schedule booking/conflict checks, recovery verification, and acknowledgement escalation. See [ROUND2_SYSTEM_ARCHITECTURE.md](ROUND2_SYSTEM_ARCHITECTURE.md) for the diagram, request flow, and API inventory.

## Security and local binding

Node binds to `127.0.0.1`, so the demo is local-only. Do not commit `round2-node/.env`. `.env.example` contains placeholders. The event history is a local JSON file at `round2-node/data/events.json` and is ignored by Git.
