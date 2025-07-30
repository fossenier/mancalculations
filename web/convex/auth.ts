import { convexAuth } from "@convex-dev/auth/server";

import { Username } from "./Username";

export const { auth, signIn, signOut, store, isAuthenticated } = convexAuth({
  providers: [Username],
});
