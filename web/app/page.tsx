import { Button, Center, Flex, Group } from "@mantine/core";

export default function Home() {
  return (
    <>
      <main>
        <Center sx={{ minHeight: "100vh" }}>
          <Flex align="center" direction="column">
            <h1>Mancala</h1>
            <Group mt="xl" mb="md">
              <Button color="blue">Play</Button>
              <Button color="lime">Login</Button>
            </Group>
          </Flex>
        </Center>
      </main>
    </>
  );
}
