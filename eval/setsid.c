// mac os doesn't have a native setsid command
// gcc setsid.c -o setsid

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>

int main(int argc, char *argv[]) {
    pid_t pid;

    if (argc < 3) {
        fprintf(stderr, "usage: %s <session-file> <command> [args...]\n", argv[0]);
        return 1;
    }

    const char *session_file_path = argv[1];

    pid = fork();
    if (pid < 0) {
        perror("fork");
        return 1;
    }

    if (pid > 0) {
        exit(0); // Parent exits
    }

    if (setsid() < 0) {
        perror("setsid");
        return 1;
    }

    pid_t sid = getsid(0);
    if (sid < 0) {
        perror("getsid");
        return 1;
    }

    FILE *session_file = fopen(session_file_path, "w");
    if (session_file == NULL) {
        perror("fopen session file");
        return 1;
    }
    if (fprintf(session_file, "%d\n", sid) < 0) {
        perror("fprintf session file");
        fclose(session_file);
        return 1;
    }
    if (fflush(session_file) != 0) {
        perror("fflush session file");
        fclose(session_file);
        return 1;
    }
    if (fclose(session_file) != 0) {
        perror("fclose session file");
        return 1;
    }

    // Execute the command passed as arguments
    execvp(argv[2], &argv[2]);
    perror("execvp"); // Only reached if execvp fails
    return 1;
}