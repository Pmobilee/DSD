def p_sample_ddim_student(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device
       
        if self.model.parameterization == "x0":
            # Calculating v directly via the apply_model function, x=the previous clean latent
            v = self.model.apply_model(x, t, c)
            
            # Alpha and Sigma related calculations
            alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas

            # Sigmas are empty with ddim_eta = 0.0, is this correct during distillation?
            eta = 0.01
            sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

            # Selecting timestep-dependent parameters
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

            # Creating the v-prediction
            pred = (x * a_t - v * sqrt_one_minus_at)
            
            # a_t * e_t - sigma_t * x
            # WHAT IS EPS?
            eps = (x - a_t * pred) / sigma_t

        
            # Applying DDIM adjustments if t > 0
            if t > 0:
                # Assuming a_prev and sigma_prev for t-1 are directly computable
                a_t_prev = a_prev
                sigma_t_prev = torch.full((b, 1, 1, 1), sigmas[index-1], device=device) if index > 0 else sigma_t  # Replace with your logic if different
                
                ddim_sigma = eta * (sigma_t_prev ** 2 / sigma_t ** 2).sqrt() * \
                            (1 - a_t ** 2 / a_t_prev ** 2).sqrt()
                adjusted_sigma = (sigma_t_prev ** 2 - ddim_sigma ** 2).sqrt()
                pred = pred * a_t_prev + eps * adjusted_sigma

                if eta:
                    pred += torch.randn_like(pred) * ddim_sigma

            return pred, sigma_t, a_t, v