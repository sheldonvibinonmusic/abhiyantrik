# Round 2 Integration Report

## Current implementation

- Vanilla Mission Control frontend is served by Node at `127.0.0.1:3000`; it uses same-origin API calls and does not ask for a backend URL.
- Node validates single-log requests, calls the local Python model service, records event results, serves bounded random `test.csv` simulation, and exposes recipient/live-alert settings and SMTP test/verify routes.
- Python FastAPI loads the verified `model_artefact_verified.joblib` artifact and returns class, score, threshold, model version, and inference time.
- SMTP configuration and app password are read from `round2-node/.env`; recipient and live-alert toggle are entered in Settings and held in backend memory until restart.
- The simulation uses random test rows, labels them as synthetic/test data, defaults to dry-run unless live alert settings are configured, and respects the per-run email cap.
- Architecture and current limitations are documented in `ROUND2_SYSTEM_ARCHITECTURE.md`.

## Verification performed

- Node server and source JavaScript passed `node --check`.
- Python model API passed `py_compile`.
- Both inline frontend scripts passed `node --check`.
- Local `GET /api/health` reported Node healthy and inference available for `EXP_C_HYBRID_BLEND-threshold-0.50`.
- Local `GET /` returned HTTP 200.
- Two-record `POST /api/simulate` dry-run returned two model responses, zero model failures, and zero emails sent.
- SMTP configuration verification succeeded and a test message was accepted by SMTP during the interactive session. SMTP acceptance does not establish inbox delivery.
- No formal Node test suite is configured. `test_pipeline.py` is a model training/submission script, not a unit test suite; it was not run as part of integration verification.

## Not implemented

No GraphQL schema exists. Real crew routing, persistent vector memory, conflict-safe repair booking, recovery verification, acknowledgement escalation, and WhatsApp are not implemented. The corresponding Mission Control sections are roadmap/demo UI, not live workflow capabilities. Event history is local JSON and is not designed for concurrent multi-user deployment.

## Credentials and generated files

`round2-node/.env` is ignored and must not be committed. `round2-node/data/events.json`, `round2-node/node_modules`, `predictions.csv`, and `submission.csv` are generated/local files and are ignored. `.env.example` contains placeholders only.
