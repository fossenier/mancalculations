import { Button, Center, Flex, Group } from "@mantine/core";

export default function Home() {
  return (
    <>
      <main>
        <Center style={{ minHeight: "100vh" }}>
          <Flex align="center" direction="column">
            <h1>Mancala</h1>
            <Group mt="xl" mb="md">
              <Button size="lg" radius="md" color="blue">
                Play
              </Button>
              <Button size="lg" radius="md" color="lime">
                Login
              </Button>
            </Group>
          </Flex>
        </Center>
      </main>
    </>
  );
}
