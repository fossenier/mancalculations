"use client";

import Link from "next/link";
import { useState } from "react";

import { useAuthActions } from "@convex-dev/auth/react";
import {
  Alert,
  Button,
  Center,
  Flex,
  Paper,
  PasswordInput,
  Text,
  TextInput,
  Title,
} from "@mantine/core";
import { useForm } from "@mantine/form";
import { IconAlertCircle } from "@tabler/icons-react";

interface LoginFormValues {
  username: string;
  password: string;
}

export default function Login() {
  const { signIn } = useAuthActions();
  // const [error, setError] = useState<string | null>(null);
  // const [isLoading, setIsLoading] = useState(false);

  // Temporarily replace your form with this exact example
  return (
    <main>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          const formData = new FormData(event.currentTarget);
          void signIn("password", formData);
        }}
      >
        <input name="email" placeholder="Email" type="text" />
        <input name="password" placeholder="Password" type="password" />
        <input name="flow" type="hidden" value="signIn" />
        <button type="submit">Sign in</button>
      </form>
    </main>
  );

  // const form = useForm({
  //   mode: "uncontrolled",
  //   initialValues: {
  //     username: "",
  //     password: "",
  //   },
  //   validate: {
  //     username: (value) =>
  //       value.length < 3 ? "Username must be at least 3 characters" : null,
  //     password: (value) =>
  //       value.length < 6 ? "Password must be at least 6 characters" : null,
  //   },
  // });

  // const handleSubmit = async (values: LoginFormValues): Promise<void> => {
  //   setError(null);
  //   setIsLoading(true);

  //   try {
  //     const formData = new FormData();
  //     formData.append("email", values.username); // Using username as email for Convex Auth
  //     formData.append("password", values.password);
  //     formData.append("flow", "signIn");

  //     // Debug: Log what we're sending
  //     console.log("Attempting login with:", {
  //       email: values.username,
  //       flow: "signIn",
  //     });

  //     const result = await signIn("password", formData);
  //     console.log("Sign in result:", result);

  //     // Convex Auth will handle the redirect after successful login
  //   } catch (error: unknown) {
  //     console.error("Login failed:", error);

  //     // Better error handling
  //     if (error instanceof Error) {
  //       if (error.message.includes("<!DOCTYPE")) {
  //         setError(
  //           "Authentication service is not properly configured. Please check your Convex Auth setup."
  //         );
  //       } else if (error.message.includes("Invalid credentials")) {
  //         setError("Invalid username or password.");
  //       } else {
  //         setError(
  //           error.message || "An unexpected error occurred during login."
  //         );
  //       }
  //     } else {
  //       setError("An unexpected error occurred. Please try again.");
  //     }
  //   } finally {
  //     setIsLoading(false);
  //   }
  // };

  // return (
  //   <main>
  //     <Center style={{ minHeight: "100vh" }}>
  //       <Flex align="center" direction="column">
  //         <Title order={1} mb="xl">
  //           iSE Mancala
  //         </Title>

  //         <Paper
  //           withBorder
  //           shadow="md"
  //           p={30}
  //           mt={30}
  //           radius="md"
  //           style={{ minWidth: 350 }}
  //         >
  //           <Title order={2} ta="center" mb="md">
  //             Welcome back!
  //           </Title>
  //           <Text c="dimmed" size="sm" ta="center" mb="lg">
  //             Login to your account to continue playing
  //           </Text>

  //           {error && (
  //             <Alert
  //               icon={<IconAlertCircle size={16} />}
  //               title="Login failed"
  //               color="red"
  //               mb="md"
  //               withCloseButton
  //               onClose={() => setError(null)}
  //             >
  //               {error}
  //             </Alert>
  //           )}

  //           <form onSubmit={form.onSubmit(handleSubmit)}>
  //             <TextInput
  //               withAsterisk
  //               label="Username"
  //               placeholder="Enter your username"
  //               key={form.key("username")}
  //               {...form.getInputProps("username")}
  //               mb="md"
  //               disabled={isLoading}
  //             />

  //             <PasswordInput
  //               withAsterisk
  //               label="Password"
  //               placeholder="Enter your password"
  //               key={form.key("password")}
  //               {...form.getInputProps("password")}
  //               mb="lg"
  //               disabled={isLoading}
  //             />

  //             <Button
  //               type="submit"
  //               fullWidth
  //               size="md"
  //               radius="md"
  //               color="lime"
  //               loading={isLoading}
  //               disabled={isLoading}
  //             >
  //               {isLoading ? "Logging in..." : "Login"}
  //             </Button>
  //           </form>

  //           <Button
  //             component={Link}
  //             href="/"
  //             variant="subtle"
  //             size="sm"
  //             color="blue"
  //             fullWidth
  //             mt="lg"
  //             disabled={isLoading}
  //           >
  //             ← Back to Home
  //           </Button>
  //         </Paper>
  //       </Flex>
  //     </Center>
  //   </main>
  // );
}
