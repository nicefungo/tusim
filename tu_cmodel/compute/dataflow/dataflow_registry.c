/*
 * TU Dataflow Registry — Implementation
 * ======================================
 */

#include "dataflow_registry.h"
#include <stdlib.h>
#include <string.h>

#define TU_DATAFLOW_MAX_PLUGINS 8

static tu_dataflow_plugin_t *g_registry[TU_DATAFLOW_MAX_PLUGINS] = {NULL};
static int g_registry_count = 0;

void tu_dataflow_registry_init(void) {
    /* Already initialized? do nothing. */
}

void tu_dataflow_register(tu_dataflow_plugin_t *plugin) {
    if (!plugin) return;
    if (g_registry_count >= TU_DATAFLOW_MAX_PLUGINS) return;

    /* Check for duplicate ID */
    for (int i = 0; i < g_registry_count; i++) {
        if (g_registry[i] && g_registry[i]->id == plugin->id) {
            /* Replace existing */
            if (g_registry[i]->impl_data) free(g_registry[i]->impl_data);
            free(g_registry[i]);
            g_registry[i] = plugin;
            return;
        }
    }

    g_registry[g_registry_count++] = plugin;
}

tu_dataflow_plugin_t *tu_dataflow_lookup(tu_dataflow_id_t id) {
    for (int i = 0; i < g_registry_count; i++) {
        if (g_registry[i] && g_registry[i]->id == id)
            return g_registry[i];
    }
    return NULL;
}

tu_dataflow_plugin_t *tu_dataflow_lookup_by_name(const char *name) {
    if (!name) return NULL;
    for (int i = 0; i < g_registry_count; i++) {
        if (g_registry[i] && g_registry[i]->name &&
            strcmp(g_registry[i]->name, name) == 0)
            return g_registry[i];
    }
    return NULL;
}

int tu_dataflow_registry_count(void) {
    return g_registry_count;
}

tu_dataflow_plugin_t *tu_dataflow_registry_get(int index) {
    if (index < 0 || index >= g_registry_count) return NULL;
    return g_registry[index];
}

void tu_dataflow_registry_destroy(void) {
    for (int i = 0; i < g_registry_count; i++) {
        if (g_registry[i]) {
            if (g_registry[i]->impl_data) free(g_registry[i]->impl_data);
            free(g_registry[i]);
            g_registry[i] = NULL;
        }
    }
    g_registry_count = 0;
}
