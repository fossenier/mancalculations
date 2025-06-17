"use client";
import { useMutation } from "convex/react";
import { ConvexError } from "convex/values";
import { useRouter } from "next/navigation";
import React, { useState } from "react";

import { api } from "@/convex/_generated/api";
import { Button, TextField } from "@mui/material";

export default function Login() {
  // Redirection once logged in
  const router = useRouter();

  // Form controlled user inputs
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  // Error messages on user input (from server)
  const [usernameError, setUsernameError] = useState("");
  const [passwordError, setPasswordError] = useState("");
  const [buttonError, setButtonError] = useState("");

  // Client side username validation (server side exists)
  const isUsernameValid = (): boolean => {
    if (username.length >= 6) {
      setUsernameError("");
      return true;
    } else {
      setUsernameError("Username must be 6 or more characters");
      return false;
    }
  };

  // Client side password validation (server side exists)
  const isPasswordValid = (): boolean => {
    if (password.length >= 8) {
      setPasswordError("");
      return true;
    } else {
      setPasswordError("Password must be 8 or more characters");
      return false;
    }
  };

  // The loginUser Convex mutation
  const loginUser = useMutation(api.mutations.userAuthentication.loginUser);

  const handleLogin = async (): Promise<void> => {
    // Don't call the server mutation when client side validation fails
    if (usernameError != "" || passwordError != "") {
      setButtonError("Username or password is invalid");
      return;
    }
    // Clear any previous button error if the username + password is good
    setButtonError("");

    // Login the user via Convex, handle any errors
    try {
      const sessionId = (await loginUser({
        username,
        password,
      })) as string;
      if (sessionId) {
        const response = await fetch("/api/authenticateClient", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify([sessionId, username]),
        });

        if (response.ok) {
          router.push("/play");
        } else {
          setButtonError("An unknown error occurred");
        }
      }
    } catch (error) {
      if (error instanceof ConvexError) {
        const { message, serverUsernameError, serverPasswordError } =
          error.data as {
            message: string;
            serverUsernameError: boolean;
            serverPasswordError: boolean;
          };
        if (serverUsernameError) {
          // Convex sent a username related error, display it there
          setUsernameError(message);
        } else if (serverPasswordError) {
          // Convex sent a password related error, display it there
          setPasswordError(message);
        } else {
          // Convex sent an error not related necessarily to username or password
          setButtonError(message);
        }
      } else {
        // This is an unplanned scenario, display a generic error
        setButtonError("An unknown error occurred");
      }
    }
  };

  return (
    <div className="h-dvh w-dvw flex flex-row justify-center items-center bg-white">
      <div className="py-8 px-12 flex flex-col items-center gap-2 bg-periwinkle rounded-3xl">
        <p className="font-bold font-sans text-4xl text-chartreuse">Login</p>
        <TextField
          error={usernameError != ""}
          helperText={usernameError}
          id="username"
          label="Username"
          onChange={(e) => setUsername(e.target.value)}
          onBlur={isUsernameValid}
          value={username}
          variant="outlined"
        ></TextField>
        <TextField
          error={passwordError != ""}
          helperText={passwordError}
          id="password"
          label="Password"
          onChange={(e) => setPassword(e.target.value)}
          onBlur={isPasswordValid}
          type="password"
          value={password}
          variant="outlined"
          className="flex-1"
        ></TextField>
        <p className="text-red-600">{buttonError}</p>
        <Button variant="contained" onClick={handleLogin}>
          Submit
        </Button>
      </div>
    </div>
  );
}
