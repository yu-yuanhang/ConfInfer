#include <backend/confinfer_ta_math.h>

static float ta_math_abs_f32(float x)
{
    return x < 0.0f ? -x : x;
}

float ta_math_neg_inf_f32(void)
{
    return -3.402823466e+38f;
}

float ta_math_exp_f32(float x)
{
    static const float k_ln2 = 0.69314718056f;
    static const float k_inv_ln2 = 1.44269504089f;
    int k = 0;
    float r = 0.0f;
    float term = 1.0f;
    float sum = 1.0f;
    float scale = 1.0f;
    unsigned i = 0;

    if (x > 80.0f) {
        x = 80.0f;
    } else if (x < -80.0f) {
        x = -80.0f;
    }

    if (x >= 0.0f) {
        k = (int)(x * k_inv_ln2 + 0.5f);
    } else {
        k = (int)(x * k_inv_ln2 - 0.5f);
    }
    r = x - ((float)k * k_ln2);

    for (i = 1; i <= 6; ++i) {
        term *= r / (float)i;
        sum += term;
    }

    if (k >= 0) {
        while (k-- > 0) {
            scale *= 2.0f;
        }
        return sum * scale;
    }

    while (k++ < 0) {
        scale *= 0.5f;
    }
    return sum * scale;
}

float ta_math_sqrt_f32(float x)
{
    float guess = 0.0f;
    unsigned i = 0;

    if (x <= 0.0f) {
        return 0.0f;
    }

    guess = x > 1.0f ? x : 1.0f;
    for (i = 0; i < 8; ++i) {
        guess = 0.5f * (guess + x / guess);
    }

    if (ta_math_abs_f32(guess) < 1e-12f) {
        return 0.0f;
    }
    return guess;
}
