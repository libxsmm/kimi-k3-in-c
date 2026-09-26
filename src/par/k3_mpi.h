/* k3_mpi.h - the MPI side of tensor parallelism. Only this file's .c touches MPI; the
 * engine core sees k3_tp (k3.h). Without K3_MPI every call is a no-op and the engine
 * runs as a single rank. */
#ifndef K3_MPI_H
#define K3_MPI_H

#include "k3.h"

#ifdef K3_MPI
/* MPI_Init_thread (FUNNELED: only the main thread communicates) and fill k3_tp. */
int  k3_mpi_init(int *argc, char ***argv);
void k3_mpi_finalize(void);
/* Largest value of v across ranks, so a failure on any rank fails all of them. */
int  k3_mpi_max_int(int v);
/* Tear every rank down: a rank that fails alone would leave the others in a collective. */
void k3_mpi_abort(int rc);
#else
static inline int  k3_mpi_init(int *argc, char ***argv) { (void)argc; (void)argv; return 0; }
static inline void k3_mpi_finalize(void) {}
static inline int  k3_mpi_max_int(int v) { return v; }
static inline void k3_mpi_abort(int rc) { (void)rc; }
#endif

#endif /* K3_MPI_H */
