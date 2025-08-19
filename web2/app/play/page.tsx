import { Container, Text, Title } from "@mantine/core";

export default function PlayPage() {
  return (
    <Container size="sm" py="xl">
      <Title order={1}>Hello, World!</Title>
      <Text mt="md">
        Welcome to the /play route of your Next.js app using Mantine.
      </Text>
    </Container>
  );
}
