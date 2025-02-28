import { createClient } from '@supabase/supabase-js';

// Get the environment variables we set earlier
const supabaseUrl = import.meta.env.PUBLIC_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.PUBLIC_SUPABASE_ANON_KEY;

// Create the connection to Supabase
export const supabase = createClient(supabaseUrl, supabaseAnonKey);