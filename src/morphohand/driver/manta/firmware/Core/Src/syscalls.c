/* Minimal newlib syscall stubs -- standard boilerplate (same as what
 * STM32CubeIDE/STM32CubeMX generate) so the toolchain's libc links cleanly.
 * We don't use stdio for anything real (protocol.c talks USB-CDC directly),
 * this just satisfies the linker. */

#include <sys/stat.h>
#include <errno.h>
#include <stdint.h>

extern int _end;
static uint8_t *heap_end = (uint8_t *)&_end;

void *_sbrk(int incr)
{
    extern uint8_t _estack;
    uint8_t *prev = heap_end;
    if (heap_end + incr > &_estack) { errno = ENOMEM; return (void *)-1; }
    heap_end += incr;
    return prev;
}

int _close(int file) { (void)file; return -1; }
int _fstat(int file, struct stat *st) { (void)file; st->st_mode = S_IFCHR; return 0; }
int _isatty(int file) { (void)file; return 1; }
int _lseek(int file, int ptr, int dir) { (void)file; (void)ptr; (void)dir; return 0; }
int _read(int file, char *ptr, int len) { (void)file; (void)ptr; (void)len; return 0; }
int _write(int file, char *ptr, int len) { (void)file; (void)ptr; return len; }
void _exit(int status) { (void)status; while (1) { } }
int _kill(int pid, int sig) { (void)pid; (void)sig; errno = EINVAL; return -1; }
int _getpid(void) { return 1; }
