"""FastAPI REST layer over the MemoryService seam (Phase 1's external contract).

The desktop shell (Phase 2) will consume the service over HTTP through these routes.
The app depends only on the ``MemoryService`` ABC, so it serves either backend.
"""
