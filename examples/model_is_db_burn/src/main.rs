use burn::{
    module::Module,
    nn::{self, loss::CrossEntropyLossConfig},
    optim::AdamConfig,
    tensor::{backend::AutodiffBackend, Data, Tensor, Int},
};
use rand::Rng;
use std::collections::HashMap;
use std::time::Instant;

const VOCAB: usize = 32;
const D_MODEL: usize = 64;
const MAX_SEQ: usize = 32;
const N_HEADS: usize = 4;

#[derive(Module, Debug)]
struct FivebitModel<B: burn::tensor::backend::Backend> {
    embed: nn::Embedding<B>,
    attn: nn::MultiHeadAttention<B>,
    output: nn::Linear<B>,
}

fn create_model<B: burn::tensor::backend::Backend>(device: &B::Device) -> FivebitModel<B> {
    FivebitModel {
        embed: nn::EmbeddingConfig::new(VOCAB, D_MODEL).init(device),
        attn: nn::MultiHeadAttentionConfig::new(D_MODEL, N_HEADS)
            .with_d_k(16)
            .init(device),
        output: nn::LinearConfig::new(D_MODEL, VOCAB).init(device),
    }
}

impl<B: burn::tensor::backend::Backend> FivebitModel<B> {
    fn forward(&self, token_ids: Tensor<B, 1, Int>) -> Tensor<B, 1> {
        let seq = token_ids.dims()[0].min(MAX_SEQ);
        // Embed: [seq] → [1, seq, d_model]
        let x = self.embed.forward(token_ids.slice([0..seq]).unsqueeze());
        // Attention with causal mask
        let mask = nn::attention::generate_autoregressive_mask(seq, &x.device());
        let attended = self.attn.forward(x.clone(), x.clone(), x, Some(mask), None);
        // Pool last position → [d_model]
        let last = attended.select(1, seq - 1).squeeze(0);
        // Output: [d_model] → [vocab]
        self.output.forward(last)
    }
}

fn generate_qa(n_users: usize, n_orders: usize) -> (HashMap<usize, Vec<i32>>, Vec<(Vec<i32>, i32)>) {
    let mut rng = rand::thread_rng();
    let mut db = HashMap::new();
    for _ in 0..n_orders {
        let uid = rng.gen_range(1..=n_users);
        db.entry(uid).or_insert_with(Vec::new).push(rng.gen_range(100..50000));
    }
    let mut qa = Vec::new();
    for uid in 1..=n_users {
        let count = db.get(&uid).map(|o| o.len()).unwrap_or(0) as i32;
        // Tokenize question: start(31) + "cnt" tokens + uid digits + end(30)
        let mut q = vec![31i32, 2, 13, 19, 30]; // START C N T END
        for ch in uid.to_string().chars() {
            q.push(ch.to_digit(10).unwrap() as i32);
        }
        q.push(30); // END
        let a = (count % VOCAB as i32).max(0);
        qa.push((q, a));
    }
    (db, qa)
}

fn main() {
    type B = burn::backend::Autodiff<burn::backend::NdArray<f32>>;

    println!("{}", "═".repeat(60));
    println!("  5bit Model-IS-DB — Burn + Autodiff");
    println!("{}", "═".repeat(60));

    let device = B::Device::default();

    for (n_users, n_orders, epochs) in [(50, 200, 80), (100, 500, 60)] {
        println!("\n── {} users × {} orders ──", n_users, n_orders);
        let (_db, qa) = generate_qa(n_users, n_orders);
        let split = qa.len() / 2;

        let mut model = create_model::<B>(&device);
        let mut optim = AdamConfig::new().init::<B>();

        let t0 = Instant::now();
        for ep in 0..epochs {
            let mut total_loss = 0.0;
            for (q_tokens, a_val) in &qa[..split] {
                let input = Tensor::<B, 1, Int>::from_data(
                    Data::new(q_tokens.clone(), burn::tensor::Shape::new([q_tokens.len()])).into(),
                );
                let output = model.forward(input);

                let target = Tensor::<B, 1, Int>::from_data(
                    Data::new(vec![*a_val], burn::tensor::Shape::new([1])).into(),
                );

                let loss = CrossEntropyLossConfig::new()
                    .init(&device)
                    .forward(&output.unsqueeze::<2>(), &target);

                total_loss += loss.clone().into_scalar().to_f64();
                loss.backward();
                optim.step();
            }
            if ep % 20 == 19 {
                println!("  epoch {}: loss {:.4}", ep + 1, total_loss / split as f64);
            }
        }
        let train_t = t0.elapsed().as_secs_f64();

        let mut correct = 0;
        let test_size = qa.len() - split;
        for (q_tokens, a_val) in &qa[split..] {
            let input = Tensor::<B, 1, Int>::from_data(
                Data::new(q_tokens.clone(), burn::tensor::Shape::new([q_tokens.len()])).into(),
            );
            let output = model.forward(input);
            let pred = output.argmax(0).into_scalar().to_i32();
            if pred == *a_val { correct += 1; }
        }
        println!("  Train: {:.1}s  Acc: {:.1}%", train_t, 100.0 * correct as f64 / test_size as f64);
    }
    println!("\n═══ Burn autodiff — full backprop through attention ═══");
}
