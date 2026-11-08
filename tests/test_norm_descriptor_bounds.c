/* Descriptor span admission: public direct execution, not wrapper scratch. */
#include "tu_cmodel/compute/normalization_engine.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static unsigned checks, failures;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); } } while (0)

static int fits(uint32_t off, uint32_t count, uint32_t size) {
    return (uint64_t)off + (uint64_t)count * 4 <= size;
}

int main(int argc, char **argv) {
    int validate_only = argc > 1 && strcmp(argv[1], "--validate-only") == 0;
    const uint32_t offsets[] = {0, 1, 8, 120, 121, 124, 128, UINT32_MAX-3, UINT32_MAX};
    unsigned cases = 0;
    for (int mode=0; mode<3; ++mode) for (int inplace=0; inplace<2; ++inplace)
    for (unsigned axis_case=0; axis_case<2; ++axis_case)
    for (int field=0; field<4; ++field) for (unsigned i=0; i<sizeof offsets/sizeof offsets[0]; ++i) {
        tu_sram_region_t data, params;
        tu_sram_init_runtime(&data, 128, "data", 4, 4, 1, 2, 4);
        tu_sram_init_runtime(&params, 128, "params", 4, 4, 1, 2, 4);
        CHECK(data.banks.data && params.banks.data && data.banks.bw_banks && params.banks.bw_banks);
        if (!data.banks.data || !params.banks.data || !data.banks.bw_banks || !params.banks.bw_banks) return 2;
        memset(data.banks.data, 0, 128); memset(params.banks.data, 0, 128);
        const float x[2] = {1,3}, gamma[2] = {1,2}, beta[2] = {0,1};
        memcpy(data.banks.data+16,x,sizeof x); memcpy(params.banks.data+16,gamma,sizeof gamma);
        memcpy(params.banks.data+32,beta,sizeof beta);
        float mean[2] = {777,777}, var[2] = {777,777};
        tu_norm_desc_t d = {.mode=mode, .data_sram=&data, .data_offset=16, .elem_count=2,
            .gamma_sram=&params, .gamma_offset=16, .beta_sram=&params, .beta_offset=32,
            .epsilon=1e-5f, .norm_axis_dim=axis_case ? 2 : 0, .in_place=inplace,
            .out_offset=64, .mean_out=mode==1 ? NULL : mean, .var_out=var};
        if (field==0) d.data_offset=offsets[i];
        if (field==1) d.out_offset=offsets[i];
        if (field==2) d.gamma_offset=offsets[i];
        if (field==3) d.beta_offset=offsets[i];
        int valid = fits(d.data_offset,2,128) && (inplace || fits(d.out_offset,2,128)) &&
            fits(d.gamma_offset,2,128) && (mode!=0 || fits(d.beta_offset,2,128));
        uint8_t data_before[128], param_before[128];
        memcpy(data_before,data.banks.data,128); memcpy(param_before,params.banks.data,128);
        int admitted=tu_norm_validate_desc(&d);
        CHECK(admitted==valid);
        /* Execute negative cases only. On the old source, use --validate-only
           because wrapped offsets could induce unsafe or huge accesses. */
        if (!validate_only && !valid) {
            uint64_t result=tu_norm_execute(&d);
            CHECK(result==UINT64_MAX);
            CHECK(!memcmp(data_before,data.banks.data,128));
            CHECK(!memcmp(param_before,params.banks.data,128));
            CHECK(data.banks.reads==0 && data.banks.writes==0 && data.banks.stall_cycles==0);
            CHECK(params.banks.reads==0 && params.banks.writes==0 && params.banks.stall_cycles==0);
            CHECK(mean[0]==777 && mean[1]==777 && var[0]==777 && var[1]==777);
        }
        ++cases;
        tu_sram_destroy(&params); tu_sram_destroy(&data);
    }
    /* Valid end-to-end numerical control with separate parameter storage. */
    for (int mode=0; mode<3; ++mode) for (int inplace=0; inplace<2; ++inplace) {
        tu_sram_region_t data, params;
        tu_sram_init_runtime(&data,128,"data",4,4,1,2,4);
        tu_sram_init_runtime(&params,128,"params",4,4,1,2,4);
        const float x[2]={1,3}, g[2]={1,2}, b[2]={0,1};
        memcpy(data.banks.data,x,sizeof x); memcpy(params.banks.data+120,g,sizeof g);
        memcpy(params.banks.data+112,b,sizeof b);
        float mean[1]={777}, var[1]={777};
        tu_norm_desc_t d={.mode=mode,.data_sram=&data,.elem_count=2,
            .gamma_sram=&params,.gamma_offset=120,.beta_sram=&params,.beta_offset=112,
            .epsilon=1e-5f,.norm_axis_dim=2,.in_place=inplace,.out_offset=120,
            .mean_out=mode==1 ? NULL : mean,.var_out=var};
        CHECK(tu_norm_execute(&d)!=UINT64_MAX);
        float out[2]; memcpy(out,data.banks.data+(inplace ? 0 : 120),sizeof out);
        double den=sqrt((mode == 1 ? 5.0 : 1.0)+1e-5);
        CHECK(isfinite(out[0]) && fabs(out[0]-(mode == 1 ? 1.0/den : -1.0/den))<1e-6);
        CHECK(isfinite(out[1]) && fabs(out[1]-(mode == 1 ? 6.0/den : 2.0/den+(mode == 0 ? 1.0 : 0.0)))<1e-6);
        CHECK(fabs(var[0]-(mode == 1 ? 5.0 : 1.0))<1e-6);
        if (mode != 1) CHECK(fabs(mean[0]-2.0)<1e-6);
        tu_sram_destroy(&params); tu_sram_destroy(&data);
    }
    /* Ignored fields remain ignored during actual execution, not only validation. */
    for (int mode=0; mode<3; ++mode) {
        tu_sram_region_t data;
        tu_sram_init_runtime(&data,128,"data",4,4,1,2,4);
        const float x[2]={1,3}; memcpy(data.banks.data,x,sizeof x);
        tu_norm_desc_t d={.mode=mode,.data_sram=&data,.elem_count=2,
            .epsilon=1e-5f,.norm_axis_dim=2,.in_place=true,.out_offset=UINT32_MAX};
        if (mode != TU_NORM_LAYER_NORM) {
            d.beta_sram=&data; d.beta_offset=UINT32_MAX;
        }
        CHECK(tu_norm_validate_desc(&d));
        CHECK(tu_norm_execute(&d)!=UINT64_MAX);
        float out[2]; memcpy(out,data.banks.data,sizeof out);
        CHECK(isfinite(out[0]) && isfinite(out[1]));
        tu_sram_destroy(&data);
    }
    /* Large count arithmetic and parameter extent use the axis, not all rows. */
    {
        tu_sram_region_t data, params;
        tu_sram_init_runtime(&data,128,"data",4,4,1,2,4);
        tu_sram_init_runtime(&params,128,"params",4,4,1,2,4);
        tu_norm_desc_t d={.mode=TU_NORM_LAYER_NORM,.data_sram=&data,
            .elem_count=UINT32_C(0x40000001),.in_place=true,.epsilon=1e-5f};
        CHECK(!tu_norm_validate_desc(&d));
        d.elem_count=4; d.norm_axis_dim=2; d.gamma_sram=&params; d.gamma_offset=120;
        CHECK(tu_norm_validate_desc(&d));
        d.gamma_offset=121; CHECK(!tu_norm_validate_desc(&d));
        d.gamma_sram=NULL; d.beta_sram=&params; d.beta_offset=120;
        CHECK(tu_norm_validate_desc(&d));
        d.beta_offset=121; CHECK(!tu_norm_validate_desc(&d));
        tu_sram_destroy(&params); tu_sram_destroy(&data);
    }
    printf("normalization descriptor bounds: %u cases, %u checks, %u failures (%s)\n",
           cases,checks,failures,validate_only ? "validation only" : "full");
    return failures ? 1 : 0;
}
