/* fivebit_lib.c — shared-lib codec entry points for ctypes binding.
 * Build: gcc -O2 -shared -fPIC -o libfivebit.so fivebit_lib.c
 */
#include <stdint.h>
#include <string.h>

enum { T_D0=0, T_N1=17, T_POW=26, T_SCALE=27, T_RECORD=28, T_END=30, T_START=31 };

/* pack 5-bit tokens -> bytes (big-endian, MSB-first) */
static int pack(const uint8_t*tk,int n,uint8_t*out){
    uint32_t acc=0; int nbits=0,len=0;
    for(int i=0;i<n;i++){ acc=(acc<<5)|(tk[i]&0x1F); nbits+=5; while(nbits>=8){ nbits-=8; out[len++]=(uint8_t)((acc>>nbits)&0xFF);} }
    if(nbits>0) out[len++]=(uint8_t)((acc<<(8-nbits))&0xFF);
    return len;
}
static void enc_int(long long value, uint8_t*tk, int*n){
    if(value==0){ tk[(*n)++]=T_D0; tk[(*n)++]=T_END; return; }
    int neg=value<0; unsigned long long a=neg?(unsigned long long)(-(value+1))+1ULL:(unsigned long long)value;
    char d[32]; int k=0; while(a){ d[k++]=(char)('0'+a%10); a/=10; }
    for(int i=k-1;i>=0;i--){ int dig=d[i]-'0'; tk[(*n)++]= dig==0?T_D0:(uint8_t)(neg?16+dig:dig); }
    tk[(*n)++]=T_END;
}

/* exported: encode two ints + RECORD, pack. returns byte length; writes *bit_len */
int fb_encode_two_ints(long long a, long long b, uint8_t*out, int*bit_len){
    uint8_t tk[64]; int n=0;
    enc_int(a,tk,&n); enc_int(b,tk,&n); tk[n++]=T_RECORD;
    *bit_len = n*5;
    return pack(tk,n,out);
}

/* exported: unpack+parse, extract integer values. returns count of ints. */
int fb_decode_ints(const uint8_t*data,int nbytes,int pad,long long*out,int max){
    int total_bits=nbytes*8-pad; int ntok=total_bits/5;
    /* unpack */
    uint8_t tok[256]; uint32_t acc=0; int nbits=0,produced=0;
    for(int i=0;i<nbytes && produced<ntok;i++){ acc=(acc<<8)|data[i]; nbits+=8; while(nbits>=5&&produced<ntok){ nbits-=5; tok[produced++]=(acc>>nbits)&0x1F; } }
    /* parse NUM-state integers (records are ints + RECORD) */
    int st=0; long long acc_d[64]; int nd=0; int ni=0;
    for(int i=0;i<ntok;i++){ int t=tok[i];
        if(st==0){
            if(t==T_START){ if(nd){ long long v=0; for(int j=0;j<nd;j++) v=v*10+acc_d[j]; if(ni<max)out[ni++]=v; nd=0;} st=1; }
            else if(t==T_END||t==T_RECORD){ if(nd){ long long v=0; for(int j=0;j<nd;j++) v=v*10+acc_d[j]; if(ni<max)out[ni++]=v; nd=0;} }
            else if(t<=9){ acc_d[nd++]=t; }
            else if(t>=T_N1&&t<=25){ acc_d[nd++]=-(t-16); }
        } else { /* skip word contexts for this int-only fast path */
            if(t==T_END) st=0; else if(t==T_START) st=2; else if(t==T_RECORD) st=0;
            if(st==2){ if(t==T_END) st=1; else if(t==T_RECORD) st=0; }
        }
    }
    if(nd){ long long v=0; for(int j=0;j<nd;j++) v=v*10+acc_d[j]; if(ni<max)out[ni++]=v; }
    return ni;
}
