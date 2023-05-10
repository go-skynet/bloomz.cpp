#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

void *bloomz_allocate_state();

int bloomz_bootstrap(const char *model_path, void *state_pr, int n_ctx, bool f16memory);

void* bloomz_allocate_params(const char *prompt, int seed, int threads, int tokens,
                            int top_k, float top_p, float temp, float repeat_penalty, int repeat_last_n);

void bloomz_free_params(void* params_ptr);
void bloomz_free_model(void* params_ptr);

int bloomz_predict(void* params_ptr, void* state_pr, char* result);

#ifdef __cplusplus
}
#endif
