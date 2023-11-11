#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Constants
#define HASH_TABLE_SIZE 100000
#define MAX_STONES 48
#define NUM_PITS 14
#define PLAYER1 0
#define PLAYER2 1
#define PLAYER_PIT_COUNT 6
#define INITIAL_STONES 4 // Initial number of stones in each pit

// Structs
typedef struct GameState
{
    uint8_t pits[NUM_PITS];
    struct GameState *children[PLAYER_PIT_COUNT];
} GameState;

typedef struct HashTableEntry
{
    GameState *state;
    struct HashTableEntry *next;
} HashTableEntry;

// Function Declarations
unsigned long hash(GameState *state);
void initHashTable();
void insertGameState(GameState *state);
void deleteHashTable();
int isInTree(GameState *state);
GameState *createGameState();
int checkGameOver(uint8_t pits[NUM_PITS]);
void moveAndStealRocks(GameState *state, int player_turn, int pit_selection);
void generateChildren(GameState *state, int player_turn);
void simulateGame(GameState *state, int player_turn);
void deleteGameTree(GameState *state);

// Global Variables
HashTableEntry *hashTable[HASH_TABLE_SIZE];

// Main function
int main()
{
    initHashTable();
    GameState *root = createGameState();
    if (!root)
    {
        fprintf(stderr, "Failed to create initial game state.\n");
        return EXIT_FAILURE;
    }

    simulateGame(root, PLAYER1);
    deleteHashTable();
    deleteGameTree(root);

    return 0;
}

// Function Implementations

// Hash function to generate a unique key for a game state
unsigned long hash(GameState *state)
{
    unsigned long hashValue = 0;
    // A larger prime number for the base
    const unsigned long prime = 5381;
    for (int i = 0; i < NUM_PITS; ++i)
    {
        // Bitwise left shift and xor to mix bits
        hashValue = (hashValue << 5) ^ (hashValue >> 27) ^ (state->pits[i]);
    }
    return hashValue;
}

// Initializes the hash table with NULL entries
void initHashTable()
{
    for (int i = 0; i < HASH_TABLE_SIZE; ++i)
    {
        hashTable[i] = NULL;
    }
}

// Inserts a new game state into the hash table
void insertGameState(GameState *state)
{
    unsigned long hashValue = hash(state) % HASH_TABLE_SIZE;
    HashTableEntry *entry = (HashTableEntry *)malloc(sizeof(HashTableEntry));
    if (!entry)
    {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
    entry->state = state;
    entry->next = hashTable[hashValue];
    hashTable[hashValue] = entry;
}

// Deletes the hash table and frees memory
void deleteHashTable()
{
    for (int i = 0; i < HASH_TABLE_SIZE; ++i)
    {
        HashTableEntry *entry = hashTable[i];
        while (entry != NULL)
        {
            HashTableEntry *temp = entry;
            entry = entry->next;
            free(temp->state); // Free the game state
            free(temp);        // Free the hash table entry
        }
        hashTable[i] = NULL;
    }
}

// Checks if a given game state already exists in the hash table
int isInTree(GameState *state)
{
    unsigned long hashValue = hash(state) % HASH_TABLE_SIZE;
    HashTableEntry *entry = hashTable[hashValue];
    while (entry != NULL)
    {
        if (memcmp(entry->state->pits, state->pits, NUM_PITS) == 0)
        {
            return 1; // The game state is found in the hash table
        }
        entry = entry->next;
    }
    return 0; // The game state is not found
}

// Creates a new game state with the initial configuration
GameState *createGameState()
{
    GameState *state = (GameState *)malloc(sizeof(GameState));
    if (!state)
    {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < NUM_PITS; ++i)
    {
        state->pits[i] = (i % 7 == 6) ? 0 : INITIAL_STONES;
    }
    for (int i = 0; i < PLAYER_PIT_COUNT; ++i)
    {
        state->children[i] = NULL;
    }
    return state;
}

// Checks if the game is over and handles capturing remaining stones
int checkGameOver(uint8_t pits[NUM_PITS])
{
    for (int player = 0; player < 2; ++player)
    {
        int sum = 0;
        int start_pit = player * (PLAYER_PIT_COUNT + 1);
        for (int i = 0; i < PLAYER_PIT_COUNT; ++i)
        {
            sum += pits[start_pit + i];
        }
        if (sum == 0)
        {
            // capture the remaining stones for the other player
            int other_player = 1 - player;
            int other_start_pit = other_player * (PLAYER_PIT_COUNT + 1);
            for (int i = 0; i < PLAYER_PIT_COUNT; ++i)
            {
                pits[other_start_pit + PLAYER_PIT_COUNT] += pits[other_start_pit + i];
                pits[other_start_pit + i] = 0;
            }
            return 1;
        }
    }
    return 0;
}

// Performs a move and handles the stealing of stones if applicable
void moveAndStealRocks(GameState *state, int player_turn, int pit_selection)
{
    // move rocks and possibly get an extra turn
    int rock_count = state->pits[pit_selection];
    state->pits[pit_selection] = 0;
    int current_pit = pit_selection;

    while (rock_count > 0)
    {
        current_pit = (current_pit + 1) % NUM_PITS;

        // skip opponent's store
        if (player_turn == PLAYER1 && current_pit == 13)
            continue;
        if (player_turn == PLAYER2 && current_pit == 6)
            continue;

        // place one rock in the current pit
        state->pits[current_pit]++;
        rock_count--;

        // if this was the last rock...
        if (rock_count == 0)
        {
            // check for a steal
            if (current_pit / 7 == player_turn && state->pits[current_pit] == 1)
            {
                int opposite_pit = 12 - current_pit;
                if (state->pits[opposite_pit] > 0)
                {
                    // perform steal
                    state->pits[player_turn * 7 + 6] += state->pits[opposite_pit] + 1; // Add to store
                    state->pits[opposite_pit] = 0;                                     // Clear the stolen pit
                    state->pits[current_pit] = 0;                                      // Clear the pit that triggered the steal
                }
            }

            // if the last rock lands in the player's store, they get another turn
            if ((player_turn == PLAYER1 && current_pit == 6) || (player_turn == PLAYER2 && current_pit == 13))
            {
                generateChildren(state, player_turn); // recurse with the same player's turn
                return;
            }
        }
    }

    // if no extra turn, switch player
    generateChildren(state, 1 - player_turn);
}

// Generates child game states for all possible moves from the current state
void generateChildren(GameState *state, int player_turn)
{
    if (checkGameOver(state->pits))
    {
        return; // no children if the game is over
    }

    for (int i = player_turn * 7; i < player_turn * 7 + PLAYER_PIT_COUNT; ++i)
    {
        if (state->pits[i] == 0)
            continue; // skip if the pit is empty

        GameState *newState = createGameState();
        memcpy(newState->pits, state->pits, NUM_PITS); // copy the current state

        moveAndStealRocks(newState, player_turn, i);

        if (!isInTree(newState))
        {
            insertGameState(newState);
            state->children[i - player_turn * 7] = newState;
        }
        else
        {
            free(newState); // free the state if it's already in the tree
        }
    }
}

// Simulates the game from a given state and generates the game tree recursively
void simulateGame(GameState *state, int player_turn)
{
    if (checkGameOver(state->pits))
    {
        // if the game is over, there is nothing more to do here
        return;
    }

    // generate children for the current state
    generateChildren(state, player_turn);

    // recursively simulate the game for each child
    for (int i = 0; i < PLAYER_PIT_COUNT; ++i)
    {
        if (state->children[i] != NULL)
        {
            simulateGame(state->children[i], 1 - player_turn);
        }
    }
}

// Deletes the game tree recursively and frees memory
void deleteGameTree(GameState *state)
{
    // recursively delete all child nodes
    for (int i = 0; i < PLAYER_PIT_COUNT; ++i)
    {
        if (state->children[i] != NULL)
        {
            deleteGameTree(state->children[i]);
        }
    }
    // free the current node
    free(state);
}