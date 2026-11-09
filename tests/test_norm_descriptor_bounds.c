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

/* Rectangular partitions must either process every element or reject before any
 * data, parameters, statistics, or counters are touched. */
static void check_row_partitions(int validate_only) {
    unsigned cases = 0;
    for (int mode=0; mode<3; ++mode) for (int inplace=0; inplace<2; ++inplace)
    for (uint32_t n=1; n<=12; ++n) for (uint32_t axis=0; axis<=14; ++axis) {
        tu_sram_region_t data, params;
        tu_sram_init_runtime(&data,256,"data",4,4,1,2,4);
        tu_sram_init_runtime(&params,128,"params",4,4,1,2,4);
        if (!data.banks.data || !params.banks.data || !data.banks.bw_banks || !params.banks.bw_banks) {
            fprintf(stderr,"fixture allocation failed\n"); ++failures; return;
        }
        memset(data.banks.data,0x5a,256); memset(params.banks.data,0,128);
        float input[12], g[12], b[12], means[12], vars[12];
        for (unsigned i=0; i<12; ++i) {
            input[i]=(float)(i*3+1); g[i]=1.0f+0.125f*i; b[i]=0.25f*i;
            means[i]=vars[i]=777.0f;
        }
        memcpy(data.banks.data,input,n*sizeof(float));
        memcpy(params.banks.data,g,sizeof g); memcpy(params.banks.data+64,b,sizeof b);
        tu_norm_desc_t d={.mode=mode,.data_sram=&data,.elem_count=n,
            .gamma_sram=&params,.beta_sram=&params,.beta_offset=64,.epsilon=1e-5f,
            .norm_axis_dim=axis,.in_place=inplace,.out_offset=128,
            .mean_out=mode==TU_NORM_RMS_NORM ? NULL : means,.var_out=vars};
        int valid=(axis==0 || (axis<=n && n%axis==0));
        uint8_t data_before[256], params_before[128];
        memcpy(data_before,data.banks.data,sizeof data_before);
        memcpy(params_before,params.banks.data,sizeof params_before);
        CHECK(tu_norm_validate_desc(&d)==valid);
        if (!validate_only) {
            uint64_t ret=tu_norm_execute(&d);
            if (!valid) {
                CHECK(ret==UINT64_MAX);
                CHECK(memcmp(data_before,data.banks.data,sizeof data_before)==0);
                CHECK(memcmp(params_before,params.banks.data,sizeof params_before)==0);
                CHECK(data.banks.reads==0 && data.banks.writes==0 && data.banks.stall_cycles==0);
                CHECK(params.banks.reads==0 && params.banks.writes==0 && params.banks.stall_cycles==0);
                for (unsigned i=0;i<12;++i) CHECK(means[i]==777 && vars[i]==777);
            } else {
                CHECK(ret!=UINT64_MAX);
                uint32_t width=axis ? axis : n, rows=n/width;
                float out[12]; memcpy(out,data.banks.data+(inplace ? 0 : 128),n*sizeof(float));
                for (uint32_t r=0;r<rows;++r) {
                    double mean=0, stat=0;
                    for (uint32_t j=0;j<width;++j) mean+=input[r*width+j];
                    mean/=width;
                    for (uint32_t j=0;j<width;++j) {
                        double x=input[r*width+j];
                        double t=mode==TU_NORM_RMS_NORM ? x : x-mean;
                        stat+=t*t;
                    }
                    stat/=width;
                    CHECK(fabs(vars[r]-stat)<1e-4);
                    if (mode!=TU_NORM_RMS_NORM) CHECK(fabs(means[r]-mean)<1e-5);
                    for (uint32_t j=0;j<width;++j) {
                        double x=input[r*width+j];
                        double expected=(mode==TU_NORM_RMS_NORM ? x : x-mean)/sqrt(stat+1e-5f)*g[j];
                        if (mode==TU_NORM_LAYER_NORM) expected+=b[j];
                        CHECK(isfinite(out[r*width+j]) && fabs(out[r*width+j]-expected)<1e-5);
                    }
                }
                for (uint32_t r=rows;r<12;++r) CHECK(means[r]==777 && vars[r]==777);
            }
        }
        ++cases; tu_sram_destroy(&params); tu_sram_destroy(&data);
    }
    printf("normalization row partitions: %u cases (%s)\n",cases,validate_only ? "validation only" : "full");
    /* The 2D wrappers naturally construct a rectangle; cover their public path. */
    if (!validate_only) for (int mode=0;mode<2;++mode) for (uint32_t rows=1;rows<=3;++rows)
    for (uint32_t cols=1;cols<=4;++cols) {
        tu_sram_region_t s; tu_sram_init_runtime(&s,256,"wrapper",4,4,1,2,4);
        float in[12]; for (uint32_t i=0;i<rows*cols;++i) in[i]=(float)(i+1);
        memcpy(s.banks.data,in,rows*cols*sizeof(float));
        uint64_t rc=mode ? tu_rmsnorm_2d(&s,0,rows,cols,NULL,1e-5f)
                         : tu_layernorm_2d(&s,0,rows,cols,NULL,NULL,1e-5f);
        CHECK(rc!=UINT64_MAX);
        float out[12]; memcpy(out,s.banks.data,rows*cols*sizeof(float));
        for (uint32_t r=0;r<rows;++r) {
            double mean=0, stat=0;
            for (uint32_t j=0;j<cols;++j) mean+=in[r*cols+j];
            mean/=cols;
            for (uint32_t j=0;j<cols;++j) { double x=in[r*cols+j]; double t=mode ? x : x-mean; stat+=t*t; }
            stat/=cols;
            for (uint32_t j=0;j<cols;++j) {
                double expected=(mode ? in[r*cols+j] : in[r*cols+j]-mean)/sqrt(stat+1e-5f);
                CHECK(isfinite(out[r*cols+j]) && fabs(out[r*cols+j]-expected)<1e-5);
            }
        }
        tu_sram_destroy(&s);
    }
    if (!validate_only) for (int mode=0;mode<2;++mode) for (int empty=0;empty<2;++empty) {
        tu_sram_region_t s; tu_sram_init_runtime(&s,256,"empty-wrapper",4,4,1,2,4);
        memset(s.banks.data,0x5a,256);
        uint8_t before[256]; memcpy(before,s.banks.data,sizeof before);
        uint32_t rows=empty ? 2 : 0, cols=empty ? 0 : 2;
        uint64_t rc=mode ? tu_rmsnorm_2d(&s,0,rows,cols,NULL,1e-5f)
                         : tu_layernorm_2d(&s,0,rows,cols,NULL,NULL,1e-5f);
        CHECK(rc==UINT64_MAX);
        CHECK(memcmp(before,s.banks.data,sizeof before)==0);
        CHECK(s.banks.reads==0 && s.banks.writes==0 && s.banks.stall_cycles==0);
        tu_sram_destroy(&s);
    }
}

int main(int argc, char **argv) {
    int validation_only = argc > 1 && strcmp(argv[1], "--validate-only") == 0;
    check_row_partitions(validation_only);
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
