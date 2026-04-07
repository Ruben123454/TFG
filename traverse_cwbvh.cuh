#include <cuda_runtime.h>
#include <stdint.h>
#include "helper_math.h"
#define STACK_SIZE 64

#define CWBVH_COMPRESSED_TRIS 

__device__ inline uchar4 as_uchar4(const float f) {
    uint32_t v = __float_as_uint(f);
    return make_uchar4(v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF);
}

__device__ inline char4 as_char4(const float f) {
    uint32_t v = __float_as_uint(f);
    return make_char4((char)(v & 0xFF), (char)((v >> 8) & 0xFF), (char)((v >> 16) & 0xFF), (char)((v >> 24) & 0xFF));
}

__device__ inline float4 convert_float4(const uchar4 v) {
    return make_float4((float)v.x, (float)v.y, (float)v.z, (float)v.w);
}

__device__ inline uint32_t __bfind(const uint32_t v) {
    uint32_t b;
    asm volatile("bfind.u32 %0, %1; " : "=r"(b) : "r"(v));
    return b;
}

__device__ inline float fmin_fmin(const float a, const float b, const float c) {
    return fminf(fminf(a, b), c);
}

__device__ inline float fmax_fmax(const float a, const float b, const float c) {
    return fmaxf(fmaxf(a, b), c);
}

__device__ inline uint32_t sign_extend_s8x4(const uint32_t i) {
    uint32_t v;
    asm("prmt.b32 %0, %1, 0x0, 0x0000BA98;" : "=r"(v) : "r"(i)); 
    return v;
}

#define STACK_POP(X) { X = stack[--stackPtr]; }
#define STACK_PUSH(X) { stack[stackPtr++] = X; }

#define UPDATE_HITMASK  asm( "vshl.u32.u32.u32.wrap.add %0,%1.b0, %2.b0, %3;" : "=r"(hitmask) : "r"(child_bits4), "r"(bit_index4), "r"(hitmask) )
#define UPDATE_HITMASK0 asm( "vshl.u32.u32.u32.wrap.add %0,%1.b0, %2.b0, %3;" : "=r"(hitmask) : "r"(child_bits4), "r"(bit_index4), "r"(hitmask) )
#define UPDATE_HITMASK1 asm( "vshl.u32.u32.u32.wrap.add %0,%1.b1, %2.b1, %3;" : "=r"(hitmask) : "r"(child_bits4), "r"(bit_index4), "r"(hitmask) )
#define UPDATE_HITMASK2 asm( "vshl.u32.u32.u32.wrap.add %0,%1.b2, %2.b2, %3;" : "=r"(hitmask) : "r"(child_bits4), "r"(bit_index4), "r"(hitmask) )
#define UPDATE_HITMASK3 asm( "vshl.u32.u32.u32.wrap.add %0,%1.b3, %2.b3, %3;" : "=r"(hitmask) : "r"(child_bits4), "r"(bit_index4), "r"(hitmask) )

__device__ inline float4 traverse_cwbvh(const float4* cwbvhNodes, const float4* cwbvhTris, const float3 O, const float3 D, const float3 rD, const float t_max, uint32_t* stepCount)
{
    float4 hit = make_float4(t_max, 0.0f, 0.0f, __int_as_float(-1));
    uint2 stack[STACK_SIZE];
    uint32_t hitAddr, stackPtr = 0, steps = 0;
    float2 uv;
    float tmax_current = t_max;
    
    const uint32_t octinv4 = (7 - ((D.x < 0 ? 4 : 0) | (D.y < 0 ? 2 : 0) | (D.z < 0 ? 1 : 0))) * 0x1010101;
    uint2 ngroup = make_uint2(0, 0x80000000);
    uint2 tgroup = make_uint2(0, 0);

    do {
        steps++;
        if (ngroup.y > 0x00FFFFFF) {
            const uint32_t hits = ngroup.y, imask = ngroup.y;
            const uint32_t child_bit_index = __bfind(hits);
            const uint32_t child_node_base_index = ngroup.x;
            ngroup.y &= ~(1 << child_bit_index);
            
            if (ngroup.y > 0x00FFFFFF) { STACK_PUSH(ngroup); }
            
            const uint32_t slot_index = (child_bit_index - 24) ^ (octinv4 & 255);
            const uint32_t relative_index = __popc(imask & ~(0xFFFFFFFF << slot_index));
            const uint32_t child_node_index = (child_node_base_index + relative_index) * 5;

            float4 n0 = cwbvhNodes[child_node_index + 0];
            float4 n1 = cwbvhNodes[child_node_index + 1];
            float4 n2 = cwbvhNodes[child_node_index + 2];
            float4 n3 = cwbvhNodes[child_node_index + 3];
            float4 n4 = cwbvhNodes[child_node_index + 4];

            const char4 e = as_char4(n0.w);
            ngroup.x = __float_as_uint(n1.x);
            tgroup = make_uint2(__float_as_uint(n1.y), 0);
            uint32_t hitmask = 0;

            const float idirx = __uint_as_float((e.x + 127) << 23) * rD.x;
            const float idiry = __uint_as_float((e.y + 127) << 23) * rD.y;
            const float idirz = __uint_as_float((e.z + 127) << 23) * rD.z;
            const float origx = (n0.x - O.x) * rD.x;
            const float origy = (n0.y - O.y) * rD.y;
            const float origz = (n0.z - O.z) * rD.z;

            {   
                const uint32_t meta4 = __float_as_uint(n1.z);
                const uint32_t is_inner4 = (meta4 & (meta4 << 1)) & 0x10101010;
                const uint32_t inner_mask4 = sign_extend_s8x4(is_inner4 << 3);
                const uint32_t bit_index4 = (meta4 ^ (octinv4 & inner_mask4)) & 0x1F1F1F1F;
                const uint32_t child_bits4 = (meta4 >> 5) & 0x07070707;
                
                const float4 lox4 = convert_float4(as_uchar4(rD.x < 0 ? n3.z : n2.x));
                const float4 hix4 = convert_float4(as_uchar4(rD.x < 0 ? n2.x : n3.z));
                const float4 loy4 = convert_float4(as_uchar4(rD.y < 0 ? n4.x : n2.z));
                const float4 hiy4 = convert_float4(as_uchar4(rD.y < 0 ? n2.z : n4.x));
                const float4 loz4 = convert_float4(as_uchar4(rD.z < 0 ? n4.z : n3.x));
                const float4 hiz4 = convert_float4(as_uchar4(rD.z < 0 ? n3.x : n4.z));

                float tminx0 = fmaf(lox4.x, idirx, origx), tminx1 = fmaf(lox4.y, idirx, origx);
                float tminy0 = fmaf(loy4.x, idiry, origy), tminy1 = fmaf(loy4.y, idiry, origy);
                float tminz0 = fmaf(loz4.x, idirz, origz), tminz1 = fmaf(loz4.y, idirz, origz);
                float tmaxx0 = fmaf(hix4.x, idirx, origx), tmaxx1 = fmaf(hix4.y, idirx, origx);
                float tmaxy0 = fmaf(hiy4.x, idiry, origy), tmaxy1 = fmaf(hiy4.y, idiry, origy);
                float tmaxz0 = fmaf(hiz4.x, idirz, origz), tmaxz1 = fmaf(hiz4.y, idirz, origz);
                
                n0.x = fmaxf(fmax_fmax(tminx0, tminy0, tminz0), 0.0f);
                n0.y = fminf(fmin_fmin(tmaxx0, tmaxy0, tmaxz0), tmax_current);
                n1.x = fmaxf(fmax_fmax(tminx1, tminy1, tminz1), 0.0f);
                n1.y = fminf(fmin_fmin(tmaxx1, tmaxy1, tmaxz1), tmax_current);
                if (n0.x <= n0.y) UPDATE_HITMASK;
                if (n1.x <= n1.y) UPDATE_HITMASK1;

                tminx0 = fmaf(lox4.z, idirx, origx), tminx1 = fmaf(lox4.w, idirx, origx);
                tminy0 = fmaf(loy4.z, idiry, origy), tminy1 = fmaf(loy4.w, idiry, origy);
                tminz0 = fmaf(loz4.z, idirz, origz), tminz1 = fmaf(loz4.w, idirz, origz);
                tmaxx0 = fmaf(hix4.z, idirx, origx), tmaxx1 = fmaf(hix4.w, idirx, origx);
                tmaxy0 = fmaf(hiy4.z, idiry, origy), tmaxy1 = fmaf(hiy4.w, idiry, origy);
                tmaxz0 = fmaf(hiz4.z, idirz, origz), tmaxz1 = fmaf(hiz4.w, idirz, origz);
                
                n0.x = fmaxf(fmax_fmax(tminx0, tminy0, tminz0), 0.0f);
                n0.y = fminf(fmin_fmin(tmaxx0, tmaxy0, tmaxz0), tmax_current);
                n1.x = fmaxf(fmax_fmax(tminx1, tminy1, tminz1), 0.0f);
                n1.y = fminf(fmin_fmin(tmaxx1, tmaxy1, tmaxz1), tmax_current);
                if (n0.x <= n0.y) UPDATE_HITMASK2;
                if (n1.x <= n1.y) UPDATE_HITMASK3;
            }

            {
                const uint32_t meta4 = __float_as_uint(n1.w);
                const uint32_t is_inner4 = (meta4 & (meta4 << 1)) & 0x10101010;
                const uint32_t inner_mask4 = sign_extend_s8x4(is_inner4 << 3);
                const uint32_t bit_index4 = (meta4 ^ (octinv4 & inner_mask4)) & 0x1F1F1F1F;
                const uint32_t child_bits4 = (meta4 >> 5) & 0x07070707;
                
                const float4 lox4 = convert_float4(as_uchar4(rD.x < 0 ? n3.w : n2.y));
                const float4 hix4 = convert_float4(as_uchar4(rD.x < 0 ? n2.y : n3.w));
                const float4 loy4 = convert_float4(as_uchar4(rD.y < 0 ? n4.y : n2.w));
                const float4 hiy4 = convert_float4(as_uchar4(rD.y < 0 ? n2.w : n4.y));
                const float4 loz4 = convert_float4(as_uchar4(rD.z < 0 ? n4.w : n3.y));
                const float4 hiz4 = convert_float4(as_uchar4(rD.z < 0 ? n3.y : n4.w));

                float tminx0 = fmaf(lox4.x, idirx, origx), tminx1 = fmaf(lox4.y, idirx, origx);
                float tminy0 = fmaf(loy4.x, idiry, origy), tminy1 = fmaf(loy4.y, idiry, origy);
                float tminz0 = fmaf(loz4.x, idirz, origz), tminz1 = fmaf(loz4.y, idirz, origz);
                float tmaxx0 = fmaf(hix4.x, idirx, origx), tmaxx1 = fmaf(hix4.y, idirx, origx);
                float tmaxy0 = fmaf(hiy4.x, idiry, origy), tmaxy1 = fmaf(hiy4.y, idiry, origy);
                float tmaxz0 = fmaf(hiz4.x, idirz, origz), tmaxz1 = fmaf(hiz4.y, idirz, origz);
                
                n0.x = fmaxf(fmax_fmax(tminx0, tminy0, tminz0), 0.0f);
                n0.y = fminf(fmin_fmin(tmaxx0, tmaxy0, tmaxz0), tmax_current);
                n1.x = fmaxf(fmax_fmax(tminx1, tminy1, tminz1), 0.0f);
                n1.y = fminf(fmin_fmin(tmaxx1, tmaxy1, tmaxz1), tmax_current);
                if (n0.x <= n0.y) UPDATE_HITMASK0;
                if (n1.x <= n1.y) UPDATE_HITMASK1;

                tminx0 = fmaf(lox4.z, idirx, origx), tminx1 = fmaf(lox4.w, idirx, origx);
                tminy0 = fmaf(loy4.z, idiry, origy), tminy1 = fmaf(loy4.w, idiry, origy);
                tminz0 = fmaf(loz4.z, idirz, origz), tminz1 = fmaf(loz4.w, idirz, origz);
                tmaxx0 = fmaf(hix4.z, idirx, origx), tmaxx1 = fmaf(hix4.w, idirx, origx);
                tmaxy0 = fmaf(hiy4.z, idiry, origy), tmaxy1 = fmaf(hiy4.w, idiry, origy);
                tmaxz0 = fmaf(hiz4.z, idirz, origz), tmaxz1 = fmaf(hiz4.w, idirz, origz);
                
                n0.x = fmaxf(fmax_fmax(tminx0, tminy0, tminz0), 0.0f);
                n0.y = fminf(fmin_fmin(tmaxx0, tmaxy0, tmaxz0), tmax_current);
                n1.x = fmaxf(fmax_fmax(tminx1, tminy1, tminz1), 0.0f);
                n1.y = fminf(fmin_fmin(tmaxx1, tmaxy1, tmaxz1), tmax_current);
                if (n0.x <= n0.y) UPDATE_HITMASK2;
                if (n1.x <= n1.y) UPDATE_HITMASK3;
            }
            ngroup.y = (hitmask & 0xFF000000) | (__float_as_uint(n0.w) >> 24);
            tgroup.y = hitmask & 0x00FFFFFF;
        } else {
            tgroup = ngroup;
            ngroup = make_uint2(0, 0);
        }

        while (tgroup.y != 0) {
#ifdef CWBVH_COMPRESSED_TRIS
            const uint32_t triangleIndex = __bfind(tgroup.y);
            const uint32_t triAddr = tgroup.x + triangleIndex * 4;
            const float4 T2 = cwbvhTris[triAddr + 2];
            const float transS = T2.x * O.x + T2.y * O.y + T2.z * O.z + T2.w;
            const float transD = T2.x * D.x + T2.y * D.y + T2.z * D.z;
            const float d = -transS / transD;
            tgroup.y -= 1 << triangleIndex;
            
            if (d <= 0.0f || d >= tmax_current) continue;
            
            const float4 T0 = cwbvhTris[triAddr + 0];
            const float4 T1 = cwbvhTris[triAddr + 1];
            const float3 I = make_float3(O.x + d * D.x, O.y + d * D.y, O.z + d * D.z);
            const float u = T0.x * I.x + T0.y * I.y + T0.z * I.z + T0.w;
            const float v = T1.x * I.x + T1.y * I.y + T1.z * I.z + T1.w;
            
            if (u >= 0.0f && v >= 0.0f && u + v < 1.0f) {
                uv = make_float2(u, v);
                tmax_current = d;
                hitAddr = __float_as_uint(cwbvhTris[triAddr + 3].w);
            }
#else
            const int triangleIndex = __bfind(tgroup.y);
            const int triAddr = tgroup.x + triangleIndex * 3;
            
            float4 tri0 = cwbvhTris[triAddr];
            float4 tri1 = cwbvhTris[triAddr + 1];
            float4 tri2 = cwbvhTris[triAddr + 2];
            
            const float3 e1 = make_float3(tri0.x, tri0.y, tri0.z);
            const float3 e2 = make_float3(tri1.x, tri1.y, tri1.z);
            const float3 v0 = make_float3(tri2.x, tri2.y, tri2.z);
            
            tgroup.y -= 1 << triangleIndex;
            
            const float3 r = cross(D, e1);
            const float a = dot(e2, r);
            const float f = 1.0f / a;
            const float3 s = make_float3(O.x - v0.x, O.y - v0.y, O.z - v0.z);
            const float u = f * dot(s, r);
            
            if (u < 0.0f || u > 1.0f) continue;
            
            const float3 q = cross(s, e2);
            const float v = f * dot(D, q);
            
            if (v < 0.0f || u + v > 1.0f) continue;
            
            const float d = f * dot(e1, q);
            if (d <= 0.0f || d >= tmax_current) continue;
            
            uv = make_float2(u, v);
            tmax_current = d;
            hitAddr = __float_as_uint(tri2.w);
#endif
        }

        if (ngroup.y <= 0x00FFFFFF) {
            if (stackPtr > 0) { STACK_POP(ngroup); } 
            else {
                hit = make_float4(tmax_current, uv.x, uv.y, __uint_as_float(hitAddr));
                break;
            }
        }
    } while (true);

    if (stepCount) *stepCount += steps;
    return hit;
}