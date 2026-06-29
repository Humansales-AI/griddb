/* fivebit_encode.c — C port of the 5bit encoder (integers, words, packing).
 * Goal: byte-identical output to python/binary_grid_db.py (Encoder + pack_to_bytes).
 * Build: gcc -O2 -o fivebit_encode fivebit_encode.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <time.h>

/* ---- Token values (match Token IntEnum) ---- */
enum {
    T_D0=0, /* D0..D9 = 0..9 */
    T_PLUS=10, T_MINUS=11, T_MUL=12, T_DIV=13, T_EQ=14, T_LPAREN=15, T_RPAREN=16,
    T_N1=17, /* N1..N9 = 17..25 */
    T_POW=26, T_SCALE=27,
    T_RECORD=28, T_CHECKSUM=29, T_END=30, T_START=31
};

/* token buffer helper */
typedef struct { uint8_t *t; int n, cap; } Toks;
static void tpush(Toks *b, uint8_t v){
    if (b->n >= b->cap){ b->cap = b->cap? b->cap*2 : 64; b->t = realloc(b->t, b->cap); }
    b->t[b->n++] = v;
}

/* ---- encode_integer: signed-digit tokens + END ---- */
static void encode_integer(long long value, Toks *out){
    if (value == 0){ tpush(out, T_D0); tpush(out, T_END); return; }
    int neg = value < 0;
    unsigned long long a = neg ? (unsigned long long)(-(value+1))+1ULL : (unsigned long long)value;
    char d[32]; int n=0;
    while (a){ d[n++] = (char)('0'+(a%10)); a/=10; }
    for (int i=n-1; i>=0; i--){
        int dig = d[i]-'0';
        if (dig==0) tpush(out, T_D0);
        else tpush(out, (uint8_t)(neg ? 16+dig : dig));
    }
    tpush(out, T_END);
}

/* ---- char -> token maps per context ---- */
static int word_tok(char c){            /* WORD: A-Z, space, '.' */
    if (c>='A'&&c<='Z') return c-'A';   /* A=0..Z=25 */
    if (c==' ') return T_POW;           /* 26 */
    if (c=='.') return T_SCALE;         /* 27 */
    return -1;
}
static int special_tok(char c){         /* SPECIAL: a-z, '@', '-' */
    if (c>='a'&&c<='z') return c-'a';   /* a=0..z=25 */
    if (c=='@') return T_POW;           /* 26 */
    if (c=='-') return T_SCALE;         /* 27 */
    return -1;
}
static int special2_tok(char c){        /* SPECIAL2: punctuation */
    static const char *S = "!\"#$%&'()*+,/:;<=>?[\\]^_`{|}";
    const char *p = strchr(S, c);
    return p ? (int)(p - S) : -1;
}

/* ---- encode_word: context-switching state machine ---- */
static int encode_word(const char *text, Toks *out){
    tpush(out, T_START);
    int depth = 0; /* 0=WORD,1=SPECIAL,2=SPECIAL2 */
    for (const char *p=text; *p; p++){
        char ch = *p; int v;
        if (isdigit((unsigned char)ch)){
            for (int k=0;k<depth;k++) tpush(out, T_END);
            depth=0;
            tpush(out, T_END);                 /* WORD->NUM */
            tpush(out, (uint8_t)(ch-'0'));      /* digit token */
            tpush(out, T_START);               /* NUM->WORD */
            continue;
        }
        if ((v=word_tok(ch))>=0){
            for (int k=0;k<depth;k++) tpush(out, T_END);
            depth=0; tpush(out,(uint8_t)v); continue;
        }
        if ((v=special_tok(ch))>=0){
            if (depth>1){ tpush(out,T_END); depth=1; }
            else if (depth<1){ tpush(out,T_START); depth=1; }
            tpush(out,(uint8_t)v); continue;
        }
        if ((v=special2_tok(ch))>=0){
            if (depth<2){ if (depth<1){ tpush(out,T_START); depth=1; } tpush(out,T_START); depth=2; }
            tpush(out,(uint8_t)v); continue;
        }
        if (isalpha((unsigned char)ch)){           /* fallback: uppercase */
            char u=(char)toupper((unsigned char)ch);
            for (int k=0;k<depth;k++) tpush(out, T_END);
            depth=0; tpush(out,(uint8_t)word_tok(u)); continue;
        }
        return -1; /* uncodeable */
    }
    for (int k=0;k<depth;k++) tpush(out, T_END);
    tpush(out, T_END);                          /* final WORD->NUM */
    return 0;
}

/* ---- pack 5-bit tokens -> bytes (big-endian, MSB-first, zero-pad tail) ---- */
static int pack(const uint8_t *tk, int n, uint8_t *out, int *pad_out){
    uint32_t acc=0; int nbits=0, len=0;
    for (int i=0;i<n;i++){
        acc = (acc<<5) | (tk[i]&0x1F);
        nbits += 5;
        while (nbits>=8){ nbits-=8; out[len++] = (uint8_t)((acc>>nbits)&0xFF); }
    }
    int pad=0;
    if (nbits>0){ out[len++] = (uint8_t)((acc << (8-nbits)) & 0xFF); pad = 8-nbits; }
    if (pad_out) *pad_out = pad;
    return len;
}

static void to_hex(const uint8_t *b, int n, char *out){
    static const char *H="0123456789abcdef";
    for (int i=0;i<n;i++){ out[2*i]=H[b[i]>>4]; out[2*i+1]=H[b[i]&0xF]; }
    out[2*n]=0;
}

/* ---- conformance mode: print hex+pad for a fixed battery ---- */
static void emit_int(long long v){
    Toks tk={0}; encode_integer(v,&tk);
    uint8_t buf[256]; int pad; int len=pack(tk.t,tk.n,buf,&pad);
    char hex[600]; to_hex(buf,len,hex);
    printf("INT\t%lld\t%s\t%d\n", v, hex, pad);
    free(tk.t);
}
static void emit_word(const char *w){
    Toks tk={0}; if (encode_word(w,&tk)!=0){ printf("WORD\t%s\tERR\t0\n", w); free(tk.t); return; }
    uint8_t buf[1024]; int pad; int len=pack(tk.t,tk.n,buf,&pad);
    char hex[2100]; to_hex(buf,len,hex);
    printf("WORD\t%s\t%s\t%d\n", w, hex, pad);
    free(tk.t);
}
static void emit_rec(long long id, long long val){
    Toks tk={0}; encode_integer(id,&tk); encode_integer(val,&tk); tpush(&tk,T_RECORD);
    uint8_t buf[256]; int pad; int len=pack(tk.t,tk.n,buf,&pad);
    char hex[600]; to_hex(buf,len,hex);
    printf("REC\t%lld,%lld\t%s\t%d\n", id, val, hex, pad);
    free(tk.t);
}

static void conformance(void){
    long long ints[]={0,1,-1,42,-42,123,-105,999999,-1000000,7,-7};
    for (size_t i=0;i<sizeof(ints)/sizeof(ints[0]);i++) emit_int(ints[i]);
    const char *ws[]={"HI","hi","Hi","a!b","a@b","a.b","user 42","id007","Alice","NewYork","A1B2"};
    for (size_t i=0;i<sizeof(ws)/sizeof(ws[0]);i++) emit_word(ws[i]);
    emit_rec(1,500); emit_rec(2,5000); emit_rec(42,-7);
}

/* ---- bench mode: encode N 2-int records; optional durable append ---- */
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec/1e9; }

static void bench(int N, int durable){
    uint8_t buf[256];
    /* pure encode+pack throughput */
    double t0=now(); long long sink=0;
    for (int i=0;i<N;i++){
        Toks tk={0}; encode_integer(i,&tk); encode_integer(i*7-3,&tk); tpush(&tk,T_RECORD);
        int pad; int len=pack(tk.t,tk.n,buf,&pad); sink+=len; free(tk.t);
    }
    double t1=now();
    double enc_s=t1-t0;
    printf("PURE_ENCODE\t%d\t%.4f\t%.0f\n", N, enc_s, N/enc_s);

    if (durable){
        FILE *f=fopen("/home/claude/out/cbench.5b","wb");
        double d0=now();
        for (int i=0;i<N;i++){
            Toks tk={0}; encode_integer(i,&tk); encode_integer(i*7-3,&tk); tpush(&tk,T_RECORD);
            int pad; int len=pack(tk.t,tk.n,buf,&pad); fwrite(buf,1,len,f); free(tk.t);
        }
        fflush(f);
        #ifdef __linux__
        fsync(fileno(f));   /* single group-commit fsync at end */
        #endif
        fclose(f);
        double d1=now(); double dur_s=d1-d0;
        printf("DURABLE_APPEND\t%d\t%.4f\t%.0f\n", N, dur_s, N/dur_s);
    }
    (void)sink;
}

#include <unistd.h>
int main(int argc, char **argv){
    if (argc>=2 && strcmp(argv[1],"conf")==0){ conformance(); return 0; }
    if (argc>=2 && strcmp(argv[1],"bench")==0){
        int N = argc>=3? atoi(argv[2]) : 100000;
        int durable = argc>=4? atoi(argv[3]) : 1;
        bench(N, durable); return 0;
    }
    fprintf(stderr,"usage: %s conf | bench [N] [durable]\n", argv[0]);
    return 1;
}
