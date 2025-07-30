import {
  DocumentByName,
  GenericDataModel,
  WithoutSystemFields,
} from "convex/server";
import { Value } from "convex/values";
import { Scrypt } from "lucia";

import {
  ConvexCredentials,
  ConvexCredentialsUserConfig,
} from "@convex-dev/auth/providers/ConvexCredentials";
import {
  createAccount,
  EmailConfig,
  GenericActionCtxWithAuthConfig,
  retrieveAccount,
} from "@convex-dev/auth/server";

export interface UsernameConfig<DataModel extends GenericDataModel> {
  /**
   * Uniquely identifies the provider, allowing to use
   * multiple different {@link Username} providers.
   */
  id?: string;
  /**
   * Perform checks on provided params and customize the user
   * information stored after sign up, including email normalization.
   *
   * Called for every flow ("signUp", "signIn", "reset",
   * "reset-verification" and "email-verification").
   */
  profile?: (
    /**
     * The values passed to the `signIn` function.
     */
    params: Record<string, Value | undefined>,
    /**
     * Convex ActionCtx in case you want to read from or write to
     * the database.
     */
    ctx: GenericActionCtxWithAuthConfig<DataModel>
  ) => WithoutSystemFields<DocumentByName<DataModel, "users">> & {
    email: string;
  };
  /**
   * Performs custom validation on password provided during sign up or reset.
   *
   * Otherwise the default validation is used (password is not empty and
   * at least 8 characters in length).
   *
   * If the provided password is invalid, implementations must throw an Error.
   *
   * @param password the password supplied during "signUp" or
   *                 "reset-verification" flows.
   */
  validatePasswordRequirements?: (password: string) => void;
  /**
   * Provide hashing and verification functions if you want to control
   * how passwords are hashed.
   */
  crypto?: ConvexCredentialsUserConfig["crypto"];
  /**
   * An Auth.js email provider used to require verification
   * before password reset.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  reset?: EmailConfig | ((...args: any) => EmailConfig);
  /**
   * An Auth.js email provider used to require verification
   * before sign up / sign in.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  verify?: EmailConfig | ((...args: any) => EmailConfig);
}

export function Username<DataModel extends GenericDataModel>(
  config: UsernameConfig<DataModel> = {}
) {
  return ConvexCredentials<DataModel>({
    id: "username",
    authorize: async (params, ctx) => {
      console.log("Authorize function called with params:", params);
      const { username, password, flow } = params;

      if (flow === "signUp") {
        const profile = config.profile?.(params, ctx) ?? defaultProfile(params);
        const { user } = await createAccount(ctx, {
          provider: "username",
          account: { id: username as string, secret: password as string },
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          profile: profile as any,
        });
        return { userId: user._id };
      }

      if (flow === "signIn") {
        const result = await retrieveAccount(ctx, {
          provider: "username",
          account: { id: username as string, secret: password as string },
        });
        if (!result) throw new Error("Invalid credentials");
        return { userId: result.user._id };
      }

      throw new Error("Invalid flow");
    },
    crypto: {
      async hashSecret(password: string) {
        return await new Scrypt().hash(password);
      },
      async verifySecret(password: string, hash: string) {
        return await new Scrypt().verify(hash, password);
      },
    },
  });
}

function defaultProfile(params: Record<string, unknown>) {
  return {
    email: params.email as string,
  };
}
