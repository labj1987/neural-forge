/*
 * The setjmp site for crate::guard, kept in C so the compiler that sees the
 * setjmp call also knows it returns twice. mingw-w64's <setjmp.h> declares
 * _setjmp with __attribute__((returns_twice)), so gcc keeps every value that is
 * live across it in memory rather than in a register the jump would clobber.
 * Rust has no such attribute; calling _setjmp from Rust only ever worked by luck
 * of register allocation.
 *
 * nf_guard_run itself returns exactly once to its Rust caller: 0 when `body`
 * returned normally, 1 when the vectored exception handler jumped back here. The
 * Rust frames between this function and the fault (the trampoline, the closure,
 * the DLL) are abandoned without unwinding.
 *
 * _setjmp is given a null frame on purpose: with a frame, mingw's longjmp unwinds
 * to it through RtlUnwindEx, which is not safe from inside a vectored handler.
 * With a null frame longjmp only restores registers.
 */
#include <setjmp.h>
#include <stddef.h>

typedef void (*nf_guard_body)(void *ctx);

size_t nf_guard_jmp_buf_size(void)
{
    return sizeof(jmp_buf);
}

int nf_guard_run(void *env, volatile int *active, nf_guard_body body, void *ctx)
{
    jmp_buf *volatile buf = (jmp_buf *)env;
    if (_setjmp(*buf, NULL) != 0) {
        *active = 0;
        return 1;
    }
    *active = 1;
    body(ctx);
    *active = 0;
    return 0;
}

__attribute__((noreturn)) void nf_guard_jump(void *env)
{
    longjmp(*(jmp_buf *)env, 1);
}
