/*
 * TU Dataflow Registry
 * =====================
 * A4: Static registry mapping tu_dataflow_id_t to dataflow plugin instances.
 *
 * The registry holds all compiled-in dataflow plugins. At initialization,
 * plugins register themselves via tu_dataflow_register(). The compute engine
 * selects a plugin via tu_dataflow_select() and dispatches MMA through it.
 *
 * Thread safety: not thread-safe (single-core cmodel). Each TU core
 * instance owns its own dataflow plugin pointer.
 */

#ifndef TU_DATAFLOW_REGISTRY_H
#define TU_DATAFLOW_REGISTRY_H

#include "dataflow_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Registry API ---- */

/* Initialize the global dataflow registry. Called once at startup. */
void tu_dataflow_registry_init(void);

/* Register a dataflow plugin. Plugins call this at program init. */
void tu_dataflow_register(tu_dataflow_plugin_t *plugin);

/* Look up a dataflow plugin by ID. Returns NULL if not registered. */
tu_dataflow_plugin_t *tu_dataflow_lookup(tu_dataflow_id_t id);

/* Look up by name (case-sensitive). Returns NULL if not found. */
tu_dataflow_plugin_t *tu_dataflow_lookup_by_name(const char *name);

/* Get the number of registered dataflow plugins. */
int tu_dataflow_registry_count(void);

/* Get the registered plugin at index (0..count-1). Returns NULL if OOB. */
tu_dataflow_plugin_t *tu_dataflow_registry_get(int index);

/* Destroy all registered plugins and clear the registry. */
void tu_dataflow_registry_destroy(void);

#ifdef __cplusplus
}
#endif

#endif /* TU_DATAFLOW_REGISTRY_H */
